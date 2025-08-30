# repair_pdf_headers.py
from pathlib import Path

def fix_pdf(path: Path) -> bool:
    data = path.read_bytes()
    i = data.find(b"%PDF-")
    if i == -1:
        print(f"[SKIP] %PDF- not found in {path}")
        return False

    # Strip any bytes before the true header
    data = data[i:]

    # Ensure proper EOF marker
    if b"%%EOF" not in data[-16:]:
        data = data.rstrip() + b"\n%%EOF\n"

    # Backup once, then overwrite
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    path.write_bytes(data)
    print(f"[FIXED] {path}")
    return True

def main():
    root = Path("data")
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        print("No PDFs under ./data")
        return
    fixed = 0
    for p in pdfs:
        try:
            if fix_pdf(p): fixed += 1
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
    print(f"\nDone. Fixed {fixed}/{len(pdfs)} PDFs.")

if __name__ == "__main__":
    main()
