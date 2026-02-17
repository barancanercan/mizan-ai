#!/usr/bin/env python3
"""
Fix script for İYİ Parti issues
Run this BEFORE starting streamlit
"""

from pathlib import Path

PROJECT_ROOT = Path("/home/baran/Desktop/Turkish-Government-Intelligence-Hub")
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

print("=" * 70)
print("🔧 İYİ PARTI FIX SCRIPT")
print("=" * 70)

# ============================================
# 1. Check PDF files
# ============================================
print("\n1️⃣ PDF Files Check:")
print("-" * 70)

if DATA_DIR.exists():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs:\n")

    for pdf in pdfs:
        print(f"  ✅ {pdf.name}")
else:
    print(f"❌ {DATA_DIR} not found!")

# ============================================
# 2. Check Vector DB folders
# ============================================
print("\n2️⃣ Vector DB Folders Check:")
print("-" * 70)

if VECTOR_DB_DIR.exists():
    dbs = sorted(VECTOR_DB_DIR.glob("*_db"))
    print(f"Found {len(dbs)} Vector DBs:\n")

    for db in dbs:
        print(f"  ✅ {db.name}")

    # Check for İYİ variants
    print("\nİYİ Parti variants:")
    variants = ["iyi_db", "İYİ_db", "IYI_db"]
    for variant in variants:
        path = VECTOR_DB_DIR / variant
        if path.exists():
            print(f"  ✅ {variant} EXISTS")
        else:
            print(f"  ❌ {variant} NOT FOUND")
else:
    print(f"❌ {VECTOR_DB_DIR} not found!")

# ============================================
# 3. Check picture files
# ============================================
print("\n3️⃣ Picture Files Check:")
print("-" * 70)

picture_dir = PROJECT_ROOT / "picture"
if picture_dir.exists():
    pngs = sorted(picture_dir.glob("*.png"))
    print(f"Found {len(pngs)} PNG files:\n")

    for png in pngs:
        size = png.stat().st_size
        print(f"  {png.name:15} ({size:6} bytes)")
else:
    print(f"❌ {picture_dir} not found!")

# ============================================
# 4. Fix İYİ Vector DB path issue
# ============================================
print("\n4️⃣ Fixing İYİ Vector DB path:")
print("-" * 70)

if VECTOR_DB_DIR.exists():
    # Check if there's an IYI_db that should be İYİ_db
    iyi_db = VECTOR_DB_DIR / "iyi_db"

    if iyi_db.exists():
        print(f"✅ Found {iyi_db}")
    else:
        print(f"❌ {iyi_db} not found")
        print("   You need to prepare data for İYİ Parti:")
        print("   python src/prepare_data.py --party İYİ")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 70)

print(
    """
If İYİ Parti is not showing:
1. Check if iyi.pdf exists in /data
2. If yes, run: python src/prepare_data.py --party İYİ
3. If no, add iyi.pdf to /data folder
"""
)
