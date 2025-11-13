# 🗳️ Politika Asistanı (Politics AI Assistant)

## 📋 Proje Özeti

Bu proje, vatandaşların siyasi partilerin görüşlerini öğrenmelerini ve karşılaştırmalarını sağlayan bir yapay zeka asistanıdır. Her siyasi parti, kendi AI asistanı ile temsil edilmektedir.

**Örnek Kullanımlar:**
- "AKP'nin ekonomi politikası nedir?"
- "CHP ve MHP'nin eğitim konusundaki görüşlerini karşılaştır"
- "Hangi parti çevre konusunda en radikal?"
- "Partilerle sohbet et, kendimi en çok hangi partiye yakın hissediyorum?"

## 🚀 Özellikler

* **Sohbet Tabanlı:** Test çözmek yerine, partilerin AI asistanları ile sohbet ederek bilgi alabilirsiniz.
* **Kişiselleştirilmiş:** Her kullanıcı kendi merak ettiği soruları sorabilir.
* **Güncel:** Sistem, partilerin yeni açıklamaları ve programları ile sürekli güncellenir.
* **Türkçe'ye Özel:** Türk siyasetini ve dilini anlayan bir yapay zeka.
* **Karşılaştırmalı:** Farklı partilerin aynı konudaki görüşlerini yan yana görebilme.

## 💻 Teknik Detaylar

### Teknoloji Stack'i
* **AI Beyin:** GeminiAPI
* **Veritabanı:** Pinecone, PostgreSQL
* **Arayüz:** Streamlit (prototip) / React + FastAPI (production) (?)
* **Deployment:** AWS EC2 (?), Docker

### Mimari
Proje, her siyasi parti için özel olarak eğitilmiş bir AI asistanı (multi-agent) ve bu asistanların parti programları, konuşmaları ve diğer resmi kaynaklardan (RAG sistemi) bilgi alarak cevap üretmesi prensibine dayanmaktadır.

## 🎯 Projenin Hedefleri

* Vatandaşların siyasi partiler hakkında daha kolay ve doğru bilgi almasını sağlamak.
* Siyasi katılımı ve farkındalığı artırmak.
* Türkiye'de AI teknolojisinin sivil alanda kullanımına öncülük etmek.
