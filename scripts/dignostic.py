#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

print("=" * 70)
print("🔍 DIAGNOSTIC: Picture Folder Check")
print("=" * 70)

picture_dir = Path("/home/baran/Desktop/mizan-ai/picture")

if not picture_dir.exists():
    print(f"❌ {picture_dir} bulunamadı!")
    exit(1)

print(f"\n✅ Found: {picture_dir}\n")

# Check files
files = sorted(picture_dir.glob("*.png"))
print(f"Found {len(files)} PNG files:\n")

for file in files:
    size = file.stat().st_size
    print(f"{file.name:15} | {size:8} bytes | ", end="")

    # Try to open as image
    try:
        img = Image.open(file)
        print(f"✅ {img.format:6} | {img.size}")
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:40]}")

print("\n" + "=" * 70)
print("İYİ Parti Check")
print("=" * 70)

# Check for İYİ variants
variants = ["iyi.png", "İYİ.png", "IYI.png"]
for variant in variants:
    path = picture_dir / variant
    print(f"{variant:15} | ", end="")
    if path.exists():
        print("✅ EXISTS")
    else:
        print("❌ NOT FOUND")

print("\n" + "=" * 70)
print("Prepared Parties Check")
print("=" * 70)

# Check data folder for PDFs
data_dir = Path("/home/baran/Desktop/mizan-ai/data")
if data_dir.exists():
    pdfs = sorted(data_dir.glob("*.pdf"))
    print(f"\nFound {len(pdfs)} PDFs in /data:\n")

    for pdf in pdfs:
        party_code = pdf.stem.upper()
        if party_code == "IYI":
            party_code = "İYİ"
        print(f"  {party_code:6} ← {pdf.name}")
else:
    print("❌ /data folder not found")

print("\n" + "=" * 70)
