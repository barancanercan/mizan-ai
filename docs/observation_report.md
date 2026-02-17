# Proje Değerlendirme ve Geliştirme Raporu

**Tarih:** 12.01.2026
**Hazırlayan:** AI Mentor

## 1. Projeye Genel Bakış

Bu proje, "Turkish Government Intelligence Hub" adıyla, Türkiye'deki siyasi partilerin programlarını ve beyannamelerini analiz etmek üzere tasarlanmış bir RAG (Retrieval-Augmented Generation) sistemidir. Proje, Streamlit arayüzü aracılığıyla kullanıcıların doğal dilde sorular sormasına ve bu sorulara partilerin dokümanlarından elde edilen bilgilerle yanıt almasına olanak tanır. Veri hazırlama sürecinde PDF dokümanları işlenmekte, parçalara ayrılmakta, gömülmekte (embedding) ve ChromaDB vektör veritabanında saklanmaktadır.

## 2. Üst Düzey Değerlendirme

Proje, konsept olarak değerli ve çalışan bir prototipe sahip. Ancak, profesyonel bir yazılım ürünü standardına ulaşabilmesi için önemli eksiklikleri bulunmaktadır. Kod tekrarı, testlerin tamamen yokluğu, konfigürasyon yönetimindeki zayıflıklar ve belgelendirme eksikliği ilk göze çarpan alanlardır. Mimari kararlar (her parti için ayrı veritabanı) küçük ölçekte çalışsa da projenin gelecekteki bakımı ve genişletilebilirliği için zorluklar teşkil etmektedir.

Özetle, proje **çalışan bir prototip** aşamasındadır ancak **üretim ortamına hazır değildir**.

## 3. Detaylı Gözlemler ve İyileştirme Önerileri

### 3.1. Mimari ve Tasarım

**Gözlem:** Her siyasi parti için (`akp`, `chp`, `iyi` vb.) ayrı bir ChromaDB veritabanı klasörü oluşturulmuş. `prepare_data.py` ve `query_system.py` bu klasör isimlerini kullanarak ilgili veritabanına bağlanıyor.

**Eleştiri:** Bu yaklaşım sürdürülebilir değildir.
- **Bakım Zorluğu:** Yeni bir parti eklendiğinde veya çıkarıldığında kodda (`config.py`, `app.py`) manuel değişiklikler yapmak gerekir. Bu, hata yapma olasılığını artırır.
- **Verimsizlik:** Tüm partiler arasında karşılaştırmalı bir arama yapmak imkansızdır. Örneğin, "Tüm partilerin eğitim konusundaki görüşleri nelerdir?" gibi bir soru bu mimariyle verimli bir şekilde cevaplanamaz.
- **Kaynak Tüketimi:** Her veritabanı kendi kaynaklarını (dosya tanıtıcıları, bellek) tüketir.

**Öneri:** **Tek bir veritabanı (Single Collection) mimarisine geçilmelidir.**
- Tüm doküman parçaları (chunks) tek bir ChromaDB koleksiyonunda saklanmalıdır.
- Her parçaya, hangi partiye ait olduğunu belirten bir **metadata** etiketi eklenmelidir. Örneğin: `{"party": "akp", "source_document": "akp_program_2023.pdf"}`.
- Sorgulama sırasında bu metadata filtre olarak kullanılabilir. Böylece hem tek bir partiye özel sorgular hem de tüm partileri kapsayan karşılaştırmalı sorgular kolayca yapılabilir. Bu değişiklik `query_system.py` ve `prepare_data.py` dosyalarında ciddi bir refaktör gerektirecektir.

### 3.2. Kod Kalitesi ve En İyi Pratikler

**Gözlem:** Kod genel olarak çalışır durumda ancak modern Python standartlarından uzak.
- **Tip İpuçları (Type Hinting):** Fonksiyonların neredeyse hiçbirinde tip ipucu (`def my_function(name: str) -> bool:`) kullanılmamış. Bu durum, kodun okunabilirliğini ve bakımını zorlaştırır, ayrıca statik analiz araçlarının hataları bulmasını engeller.
- **Docstrings:** Fonksiyonların ne işe yaradığı, hangi parametreleri aldığı ve ne döndürdüğü açıklanmamış. Sadece birkaç yerde yetersiz yorumlar var.
- **Hardcoded Değerler:** Veritabanı yolları, parti isimleri, model isimleri gibi birçok değer doğrudan kodun içine yazılmıştır (`src/app.py`, `src/prepare_data.py`).
- **Kod Tekrarı:** Streamlit arayüzünde her parti için benzer `if/elif` blokları tekrar ediyor. Bu, yeni bir parti eklemeyi hantal hale getirir.

**Öneri:**
- **Kod Formatlama ve Linting:** Projeye `black` (kod formatlayıcı) ve `ruff` (linter) entegre edilmelidir. Bu araçlar, tüm kod tabanında tutarlı bir stil sağlar ve potansiyel hataları otomatik olarak tespit eder.
- **Konfigürasyon Yönetimi:** Tüm hardcoded değerler `src/config.py` dosyasına taşınmalı ve oradan okunmalıdır. Parti listesi gibi dinamik olabilecek yapılar, `config.py` içinde tek bir liste veya dictionary olarak tanımlanmalıdır.
- **Refaktör:** `app.py`'deki `if/elif` zinciri, parti konfigürasyonunu bir döngü ile işleyen daha dinamik bir yapıya dönüştürülmelidir.
- **Docstring ve Tip İpuçları:** Projedeki tüm fonksiyonlara PEP 484 standartlarına uygun tip ipuçları ve PEP 257 standartlarına uygun docstring'ler eklenmelidir.

### 3.3. Veri Hazırlama (`prepare_data.py`)

**Gözlem:** Veri hazırlama süreci her çalıştığında tüm PDF'leri yeniden işliyor gibi görünüyor. Veritabanının zaten var olup olmadığını kontrol etse de, hangi dosyaların işlendiğini takip eden bir mekanizma yok.

**Eleştiri:** Büyük veri setlerinde bu süreç çok yavaş olacaktır. Bir dosya güncellendiğinde veya yeni bir dosya eklendiğinde tüm süreci baştan çalıştırmak verimsizdir.

**Öneri:**
- **İşlem Durumunu Takip Et:** Veritabanına veya ayrı bir log/JSON dosyasına, hangi dosyaların hangi versiyonla (örneğin dosyanın hash değeri ile) işlendiğini kaydeden bir mekanizma eklenmelidir. `prepare_data.py` çalıştığında sadece yeni veya değiştirilmiş dosyaları işlemelidir.
- **Hata Yönetimi:** PDF okuma veya işleme sırasında bir hata olursa ne olur? Script durmalı mı, devam mı etmeli? Hatalı dosyaları atlayıp raporlayan daha sağlam bir hata yönetimi (try-except blokları ve logging) eklenmelidir.

### 3.4. Test Eksikliği

**Gözlem:** Projede `tests/` klasörü ve herhangi bir test dosyası bulunmuyor.

**Eleştiri:** Bu, projenin en kritik eksikliğidir. Testler olmadan, yapılan herhangi bir değişikliğin (refaktör, yeni özellik) mevcut fonksiyonları bozmadığından emin olmak imkansızdır. Bu durum, projeyi kırılgan ve bakımı zor hale getirir.

**Öneri:**
- **`pytest` Entegrasyonu:** Projeye `pytest` test çatısı eklenmelidir.
- **Unit Testler:** `query_system.py` içindeki sorgu fonksiyonları, `prepare_data.py` içindeki metin bölme (chunking) fonksiyonları gibi saf mantık içeren bileşenler için birim testleri yazılmalıdır.
- **Entegrasyon Testleri:** Veri hazırlama ve sorgulama süreçlerinin baştan sona (PDF'den -> Vektör DB -> Sorgu -> Yanıt) çalıştığını doğrulayan entegrasyon testleri oluşturulmalıdır.

### 3.5. Belgelendirme ve Yardımcı Scriptler

**Gözlem:** `README.md` çok temel. Projenin nasıl kurulacağı, bağımlılıkların nasıl yükleneceği ve nasıl çalıştırılacağı hakkında detaylı bilgi vermiyor. `dignostic.py` ve `fix_iyi.py` gibi dosyaların ne işe yaradığı belgelenmemiş.

**Öneri:**
- **`README.md`'yi Zenginleştir:**
    - Projenin amacı ve yetenekleri.
    - Kurulum adımları (`git clone`, `pip install -r requirements.txt`).
    - Projeyi çalıştırma komutu (`streamlit run src/app.py`).
    - Proje yapısı hakkında kısa bir açıklama.
- **Scriptlerin Temizlenmesi:** `dignostic.py` ve `fix_iyi.py` gibi tek seferlik kullanılan veya artık gereksiz olan script'ler ya temizlenmeli ya da `scripts/` gibi bir klasör altına taşınarak ne işe yaradıkları belgelenmelidir.

## 4. Geliştirme Yol Haritası (Roadmap)

Yukarıdaki analiz doğrultusunda, projenin profesyonel bir seviyeye taşınması için aşağıdaki adımların izlenmesini öneriyorum.

### Faz 1: Temel İyileştirmeler ve Kod Sağlığı (Tahmini Süre: 2-3 gün)

1.  **Versiyon Kontrolü:** Mevcut tüm değişiklikleri `git`e commit'le.
2.  **Linting ve Formatlama:** Projeye `black` ve `ruff` ekle ve tüm dosyaları formatla/lint'le.
3.  **Konfigürasyon Merkezileştirme:** `src/config.py` dosyasını güçlendir. Tüm hardcoded yolları, parti listelerini, model isimlerini bu dosyaya taşı.
4.  **`README.md`'yi Güncelle:** Detaylı kurulum ve çalıştırma talimatları ekle.
5.  **Tip İpuçları ve Docstring'ler:** `src/utils.py` ve `src/config.py` gibi daha basit modüllerden başlayarak tüm projeye docstring ve tip ipuçları ekle.

### Faz 2: Mimari Refaktör ve Test Altyapısı (Tahmini Süre: 4-6 gün)

1.  **Test Altyapısını Kur:** `pytest`'i projeye ekle ve `tests/` klasörünü oluştur.
2.  **İlk Unit Testleri Yaz:** `utils.py` ve `query_system.py`'deki temel fonksiyonlar için birim testleri yaz.
3.  **Mimari Değişikliği:**
    - `prepare_data.py`'yi tek koleksiyon ve metadata kullanacak şekilde refaktör et. Bu script artık partileri `config.py`'den dinamik olarak almalı.
    - `query_system.py`'yi metadata filtrelemesini destekleyecek şekilde güncelle.
    - `app.py`'yi bu yeni mimariyle çalışacak şekilde düzenle. Parti seçimi artık sorgu için bir filtre parametresi olmalı.
4.  **Yeni Mimari İçin Testler:** Refaktör edilen veri hazırlama ve sorgulama sistemi için entegrasyon testleri yaz.

### Faz 3: Uygulama ve Özellik Geliştirme (Tahmini Süre: 3-5 gün)

1.  **Streamlit Arayüzünü İyileştir:** `app.py`'deki `if/elif` yapısını, `config.py`'deki parti listesi üzerinden dönen bir döngü ile değiştirerek dinamik hale getir.
2.  **Karşılaştırmalı Analiz Özelliği:** Arayüze "Tüm Partiler" seçeneği ekleyerek, kullanıcıların tüm veri setinde arama yapmasını sağla.
3.  **Gelişmiş Hata Yönetimi:** Streamlit arayüzünde, sorgu başarısız olduğunda veya veri bulunamadığında kullanıcıya anlamlı mesajlar göster.
4.  **Veri Yenileme Mekanizması:** `prepare_data.py`'de hangi dosyaların işlendiğini takip eden mekanizmayı kur.

### Faz 4: Otomasyon ve Dağıtım (Sürekli)

1.  **CI/CD Pipeline:** GitHub Actions veya benzeri bir araç ile her `push` işleminde testlerin, linter'ın ve formatlayıcının otomatik olarak çalışmasını sağlayan bir CI (Sürekli Entegrasyon) pipeline'ı kur.
2.  **Dockerize Etme:** Projeyi bir `Dockerfile` ile paketleyerek, bağımlılık sorunları olmadan kolayca farklı ortamlarda çalıştırılmasını sağla.
3.  **Logging:** Proje genelinde (özellikle veri hazırlama ve sorgu sisteminde) detaylı loglama mekanizması ekle.