import shutil
import zipfile
import os
from datetime import datetime
from pathlib import Path
import sys

# Proje kök dizinini ekle
sys.path.append(str(Path(__file__).parent.parent))
import src.config as config

def backup_vector_db():
    """Vector DB dizinini tarih damgalı olarak zip'ler."""
    source_dir = config.VECTOR_DB_DIR
    backup_root = Path(__file__).parent.parent / "backups"
    
    if not source_dir.exists():
        print(f"❌ Kaynak dizin bulunamadı: {source_dir}")
        return

    # Yedekleme dizinini oluştur
    backup_root.mkdir(exist_ok=True)
    
    # Dosya adı oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"vector_db_backup_{timestamp}.zip"
    backup_path = backup_root / backup_filename
    
    print(f"📦 Yedekleme başlatılıyor: {source_dir} -> {backup_path}")
    
    try:
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
                    
        print(f"✅ Yedekleme başarıyla tamamlandı: {backup_filename}")
        
        # Eski yedekleri temizle (opsiyonel - son 5 yedek kalsın)
        backups = sorted(list(backup_root.glob("vector_db_backup_*.zip")))
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                old_backup.unlink()
                print(f"🗑️ Eski yedek silindi: {old_backup.name}")
                
    except Exception as e:
        print(f"❌ Yedekleme sırasında hata oluştu: {str(e)}")

if __name__ == "__main__":
    backup_vector_db()
