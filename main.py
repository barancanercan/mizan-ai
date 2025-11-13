# main.py - YENİ LANGCHAIN VERSİYONU
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

# 1. PDF'DEN VERİ OKU
loader = PyPDFLoader("data/chp.pdf")
pages = loader.load()
text = "".join([page.page_content for page in pages])

# Metni parçalara böl (512 karakterlik parçalar)
chunks = []
for i in range(0, len(text), 512):
    chunk = text[i:i+512]
    if chunk.strip():
        chunks.append(chunk)

print(f"{len(chunks)} chunklar oluşturuldu.")

# 2. EMBEDDING MODELİNİ YÜKLE
print("----Embediing modeli yükleniyor----")
embeddings = HuggingFaceEmbeddings(
    model_name="nezahatkorkmaz/turkce-embedding-bge-m3"
)

# 3. BELGELERI VEKTÖRE ÇEVİR
print("----Chunklar Embed Ediliyor----")
chunk_embeddings = embeddings.embed_documents(chunks)

# 4. SORU SOR
question = input("Sorunuz: ")

# 5. SORUYU VEKTÖRE ÇEVİR
question_embedding = embeddings.embed_query(question)

# 6. BENZERLİK HESAPLA
similarities = np.dot(chunk_embeddings, question_embedding)

# 7. EN BENZER 3 BELGEYİ BUL
most_sim_3_index = np.argsort(similarities)[-3:][::-1]
relevant_doc = chunks[most_sim_3_index[0]]

# 8. GEMINI'YE GÖNDER
context = relevant_doc

prompt_template = PromptTemplate.from_template("""Aşağıdaki bilgilere dayanarak soruyu cevapla:

{context}

Soru: {question}
Cevap:""")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Chain oluştur
chain = prompt_template | llm | StrOutputParser()

# Çalıştır
response = chain.invoke({"context": context, "question": question})

print("\n" + response)