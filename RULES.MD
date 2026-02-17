# 📜 Project Rules & Standards

Bu kurallar, "Turkish Government Intelligence Hub" projesinin profesyonel, sürdürülebilir ve yüksek kaliteli bir şekilde geliştirilmesini sağlamak içindir.

## 🛠️ Kod Standartları

- **Tip İpuçları (Type Hinting):** Tüm fonksiyon ve metodlar PEP 484 standartlarına uygun tip ipuçları içermelidir.
- **Docstrings:** Her fonksiyon ve sınıf için PEP 257 standartlarına uygun, işlevi, parametreleri ve dönüş değerini açıklayan docstring'ler yazılmalıdır.
- **İsimlendirme:** Değişken ve fonksiyon isimleri açıklayıcı ve tutarlı (snake_case) olmalıdır.
- **Kod Formatı:** `black` formatlayıcı ve `ruff` linter kurallarına tam uyum sağlanmalıdır.

## 🏗️ Mimari Prensipler

- **Hardcoding Yasaktır:** Hiçbir sabit (dosya yolları, API anahtarları, model isimleri, renk kodları) doğrudan kod içine yazılmamalıdır; hepsi `src/config.py` içinde tanımlanmalıdır.
- **Tek Kaynak Teoremi:** Veri ve konfigürasyon tek bir merkezden yönetilmelidir (örneğin parti listesi sadece `config.py`'de güncellenmelidir).
- **Modülerlik:** UI mantığı (`app.py`), veri işleme (`prepare_data.py`) ve yapay zeka mantığı (`query_system.py`) birbirinden keskin çizgilerle ayrılmalıdır.

## 🧪 Kalite ve Güvenlik

- **Test Önceliği:** Eklenen her yeni özellik veya refaktör süreci için en az bir unit-test veya entegrasyon testi yazılmalıdır.
- **Hata Yönetimi:** Kullanıcıya asla ham Python hata mesajları gösterilmemelidir; hatalar yakalanmalı ve anlamlı/yönlendirici mesajlar verilmelidir.
- **Logging:** Kritik işlemler (veri yükleme, sorgu alma, hata durumları) `app.log` dosyasına standart bir formatla kaydedilmelidir.

## 🎨 UI/UX ve Tasarım

- **Modern Estetik:** Tasarımda "vibrant" renkler, glassmorphism efektleri ve temiz tipografi tercih edilmelidir.
- **Yanıt Hızı:** Uygulama içi etkileşimler akıcı olmalı, uzun süren işlemler için net "loading" göstergeleri kullanılmalıdır.
- **Türkçe Desteği:** Tüm kullanıcı arayüzü ve hata mesajları kusursuz bir Türkçeyle yazılmalıdır.

---

_Bu dosya projenin anayasasıdır ve her geliştirmede referans alınacaktır._
