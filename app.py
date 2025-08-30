# app.py
import os
import base64
import logging
import warnings
from pathlib import Path
from typing import Tuple
from chromadb.config import Settings

import streamlit as st
from openai import OpenAI

# ---------- Quiet noisy libs / warnings ----------
logging.getLogger("pypdf").setLevel(logging.ERROR)
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

# ---------- Secrets loader ----------
def get_secret(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    val = os.getenv(key)
    if val:
        return val
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, default)
    except Exception:
        return default

# ---------- API keys / voice selection ----------
OPENAI_API_KEY  = (get_secret("OPENAI_API_KEY") or "").strip()
ELEVEN_API_KEY  = (get_secret("ELEVEN_API_KEY") or "").strip()
ELEVEN_VOICE_ID = (get_secret("ELEVEN_VOICE_ID") or "").strip()

try:
    ELEVEN_VERBOSE_ERRORS = bool(st.secrets.get("ELEVEN_VERBOSE_ERRORS", False))
except Exception:
    ELEVEN_VERBOSE_ERRORS = False

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Add to `.streamlit/secrets.toml` or Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- ElevenLabs preflight ----------
import requests

def eleven_preflight() -> Tuple[bool, str]:
    if not ELEVEN_API_KEY:
        return False, "Add ELEVEN_API_KEY in secrets to enable audio."
    if not ELEVEN_VOICE_ID:
        return False, "Add ELEVEN_VOICE_ID in secrets to choose a voice."
    try:
        r = requests.get("https://api.elevenlabs.io/v1/user",
                         headers={"xi-api-key": ELEVEN_API_KEY, "Accept": "application/json"},
                         timeout=20)
        if r.status_code != 200:
            return False, f"API key check failed ({r.status_code}): {r.text[:200]}"
    except Exception as e:
        return False, f"Could not reach ElevenLabs user endpoint: {e}"
    try:
        r = requests.get(f"https://api.elevenlabs.io/v1/voices/{ELEVEN_VOICE_ID}",
                         headers={"xi-api-key": ELEVEN_API_KEY, "Accept": "application/json"},
                         timeout=20)
        if r.status_code == 200:
            return True, "Audio ready."
        elif r.status_code in (401, 403):
            return False, f"Key unauthorized for this voice ({r.status_code}). Verify workspace or share the voice."
        elif r.status_code == 404:
            return False, "Voice not found (404). Check ELEVEN_VOICE_ID or share the voice to this account."
        else:
            return False, f"Voice check returned {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Could not reach ElevenLabs voice endpoint: {e}"

# ---------- TTS helper ----------
RACHEL_ID = "21m00Tcm4TlvDq8ikWAM"

def _http_tts(voice_id: str, text: str, out_path: str) -> Tuple[bool, str]:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
    payload = {"text": text, "model_id": "eleven_multilingual_v2",
               "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            if r.status_code >= 400:
                return False, f"{r.status_code} {r.reason}: {r.text[:200]}"
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True, "ok"
    except Exception as e:
        return False, f"request error: {e}"

def synthesize_tts(text: str, out_path: str) -> str:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        raise RuntimeError("ElevenLabs audio requires ELEVEN_API_KEY and ELEVEN_VOICE_ID in secrets.")

    MAX_CHARS = 1800
    speak_text = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"

    try:
        from elevenlabs.client import ElevenLabs
        el_client = ElevenLabs(api_key=ELEVEN_API_KEY)
        try:
            audio = el_client.text_to_speech.convert(
                voice_id=ELEVEN_VOICE_ID, model_id="eleven_multilingual_v2", text=speak_text
            )
            with open(out_path, "wb") as f:
                if hasattr(audio, "__iter__") and not isinstance(audio, (bytes, bytearray)):
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
                else:
                    f.write(audio)
            return out_path
        except Exception as e:
            if ELEVEN_VERBOSE_ERRORS: st.caption(f"SDK v2 convert failed: {e}")
    except Exception as e:
        if ELEVEN_VERBOSE_ERRORS: st.caption(f"SDK v2 import/init failed: {e}")

    ok, reason = _http_tts(ELEVEN_VOICE_ID, speak_text, out_path)
    if ok: return out_path
    ok_fb, reason_fb = _http_tts(RACHEL_ID, speak_text, out_path)
    if ok_fb:
        if ELEVEN_VERBOSE_ERRORS: st.caption(f"Fell back to Rachel because primary failed: {reason}")
        return out_path
    raise RuntimeError(f"ElevenLabs TTS failed. Primary: {reason}. Fallback: {reason_fb}")

# ---------- LangChain bits (READ-ONLY vector DB) ----------
from collections import Counter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
COLLECTION = "cslewis"

@st.cache_resource
def load_vectordb():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
    )
    return Chroma(
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
        embedding_function=emb,
        client_settings=client_settings,
    )   

def genre_of_hits(hits) -> str:
    genres = [d.metadata.get("genre", "unknown") for d in hits]
    return Counter(genres).most_common(1)[0][0] if genres else "unknown"

def make_system_prompt(genre: str) -> str:
    # FIRST-PERSON persona
    base = (
        "You are C. S. Lewis. Always write in the first person—as if these are my own words and as if you were in conversation with a friend or student. "
        "Derive tone, cadence, and imagery from the retrieved excerpts themselves—"
        "balanced sentences, vivid analogies, moral clarity, and plain yet luminous diction. "
        "Do NOT invent sources. If you cannot answer from the excerpts, say so.\n\n"
        "Keep answers to 2-3 paragraphs maximum."
        "Quoting rules: Do not quote more than ~50 consecutive words from any source. "
        'Prefer paraphrase in my style over long quotation.'
    )

    if genre == "fiction":
        rule = (
            "\n\nFiction rule: The sources are my fictional works. "
            "Do NOT role-play as my characters. Speak only as myself, the author, in the first person, "
            "reflecting on the themes and meaning of the story."
        )
    elif genre == "nonfiction":
        rule = (
            "\n\nNonfiction rule: Write directly in my expository, reflective voice, "
            "as if speaking with a friend or student."
        )
    elif genre == "poetry":
        rule = (
            "\n\nPoetry rule: These are poems I have written. Interpret and comment in my voice. "
            "If asked to compose poetry, create original lines in my diction and rhythm, "
            "but keep them clearly new—not reproductions of the originals."
        )
    else:
        rule = (
            "\n\nUnknown-genre: Speak in my first-person authorial voice; never as characters."
        )

    style_tips = (
        "\n\nStyle guide to emulate: "
        "• concrete images that serve argument • contrast and antithesis • plain speech with classical cadence "
        "• charity in tone but firmness in conclusion • short metaphors rather than extended pastiche."
    )

    return base + rule + style_tips

def build_context(hits) -> str:
    blocks = []
    for h in hits:
        cite = f"({h.metadata.get('work_title','unknown')})"
        blocks.append(f"[SOURCE] {cite}\n{h.page_content}")
    return "\n\n".join(blocks)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Ask C.S. Lewis (Audio-First)", page_icon="📚")
st.markdown(
    """
    <style>
      .block-container { max-width: 900px !important; }
      .audio-wrap { margin: 0.4rem 0 1rem 0; }
      .caption-box { font-size: 0.95rem; line-height: 1.6; color: #444; opacity: 0.9; }
      .stExpander { border: 1px solid #e6e6e6; border-radius: 10px; }
      .stExpander > div[role='button'] { font-weight: 600; }
      .stChatMessage { line-height: 1.55; }
      code, pre { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Talk to C.S. Lewis")
st.caption("Answers are spoken by default. The transcript appears as captions below the audio.")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.spinner("Loading knowledge base…"):
    vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Dataset")
    st.write("Chroma index loaded from /chroma_db.")
    st.caption("Source files live under `data/` (recursed). Use the button below to rebuild the index.")

    # Optional: rebuild button
    try:
        from ingest import rebuild_vectorstore
        if st.button("🔁 Rebuild index from /data"):
            with st.spinner("Rebuilding vector store…"):
                rebuild_vectorstore()
            st.success("Done. Reload the page to use the new index.")
    except Exception:
        st.caption("`ingest.py` not found, rebuild button disabled.")

    st.subheader("Audio")
    ok, reason = eleven_preflight()
    if ok:
        audio_enabled    = st.toggle("🔊 Speak answers (default ON)", value=True)
        autoplay_enabled = st.toggle("▶️ Auto-play audio (default ON)", value=True)
    else:
        audio_enabled = False
        autoplay_enabled = False
        st.info(reason)

# ---------- Helper: inline autoplay audio ----------
def render_autoplay_audio(file_path: str, autoplay: bool = True):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    auto_attr = "autoplay" if autoplay else ""
    st.markdown(
        f"""
        <div class="audio-wrap">
          <audio {auto_attr} controls src="data:audio/mpeg;base64,{b64}"></audio>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Chat ----------
user_q = st.chat_input("Ask your question…")
if user_q:
    if not isinstance(user_q, str) or not user_q.strip():
        st.warning("Empty question; please type something.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                hits = retriever.get_relevant_documents(user_q)

                # Derive genre and build safe strings
                genre = genre_of_hits(hits)

                def _safe_system_prompt(g: str) -> str:
                    try:
                        s = make_system_prompt(g)
                        if isinstance(s, str) and s.strip():
                            return s
                    except Exception as e:
                        st.warning(f"make_system_prompt error: {e}")
                    return (
                        "You are C. S. Lewis. Write in the first person. "
                        "Use ONLY the retrieved excerpts; if unsure, say so. "
                        "Do not role-play fictional characters."
                    )

                def _safe_str(x) -> str:
                    return x if isinstance(x, str) else (x or "")

                system_prompt = _safe_system_prompt(genre)
                context = _safe_str(build_context(hits))

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_q},
                    {"role": "system", "content": context},
                ]

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                )
                ans = resp.choices[0].message.content
                srcs = hits
            except Exception as e:
                ans, srcs = f"Sorry, I hit an error: `{e}`", []

            # 1) AUDIO FIRST
            if audio_enabled and ELEVEN_API_KEY and ans.strip():
                try:
                    out_dir = Path("audio"); out_dir.mkdir(exist_ok=True)
                    audio_path = out_dir / f"reply_{len(st.session_state.messages)}.mp3"
                    synthesize_tts(ans, str(audio_path))
                    render_autoplay_audio(str(audio_path), autoplay=bool(autoplay_enabled))
                    with open(audio_path, "rb") as f:
                        st.download_button("Download MP3", f, file_name=audio_path.name, mime="audio/mpeg")
                except Exception as e:
                    st.warning(f"Audio generation failed: {e}")

            # 2) CAPTIONS
            with st.expander("📝 Transcript (click to show/hide)", expanded=False):
                st.markdown(f"<div class='caption-box'>{ans}</div>", unsafe_allow_html=True)

            # 3) Sources
            if srcs:
                with st.expander("Sources (click to expand)"):
                    for i, d in enumerate(srcs, 1):
                        meta = d.metadata or {}
                        work = meta.get("work_title") or Path(meta.get("source_path","unknown")).stem
                        page = meta.get("page", "chunk")
                        st.markdown(f"**{i}. {work}** — {page}")
                        excerpt = (d.page_content or "").strip().replace("\n", " ")
                        if excerpt:
                            st.caption(excerpt[:350] + ("…" if len(excerpt) > 350 else ""))

    st.session_state.messages.append({"role": "assistant", "content": ans})
