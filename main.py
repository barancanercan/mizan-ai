############################################
################ 1- IMPORT #################
############################################

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Validate API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

############################################
############# 2- READ TO PDF ###############
############################################

print("PDF yükleniyor...")
loader = PyPDFLoader("data/chp.pdf")
pages = loader.load()
text = "".join([page.page_content for page in pages])
print(f"{len(pages)} sayfa yüklendi")

############################################
################ 3- CHUNKING ###############
############################################

print("Metin chunk'lara bölünüyor...")
chunks = []
chunk_size = 512
overlap = 50

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    if chunk.strip():
        chunks.append(chunk)

print(f"{len(chunks)} chunk oluşturuldu")

############################################
########## 4- LOAD EMBEDDING MODEL #########
############################################

print("Türkçe Embedding Modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="nezahatkorkmaz/turkce-embedding-bge-m3"
)
print("Embedding modeli hazır")

############################################
############## 5- EMBEDDING ###############
############################################

print("Dökümanlar vektörize ediliyor...")
chunk_embeddings = embeddings.embed_documents(chunks)
chunk_embeddings = np.array(chunk_embeddings)  # NumPy array'e çevir
print(f"{len(chunk_embeddings)} chunk vektörize edildi")

############################################
########### 6- ASKING QUESTION #############
############################################

print("\n" + "="*60)
print("CHP Parti Tüzüğü - Soru-Cevap Sistemi")
print("="*60)
question = input("\n❓ Sorunuz: ")
print(f"Aranıyor: '{question}'")

############################################
####### 7- VECTORIZE QUESTION ##############
############################################

question_embedding = embeddings.embed_query(question)
question_embedding = np.array(question_embedding).reshape(1, -1)

############################################
######### 8- CALCULATE SIMILARITY ##########
############################################

print("Benzerlik hesaplanıyor...")
# Cosine similarity kullan (normalize edilmiş)
similarities = cosine_similarity(chunk_embeddings, question_embedding).flatten()

############################################
########## 9- MOST SIMILAR 3 DOC ###########
############################################

# En benzer 3 chunk'ı al
top_k = 3
most_similar_indices = np.argsort(similarities)[-top_k:][::-1]

relevant_chunks = [chunks[idx] for idx in most_similar_indices]
context = "\n\n".join(relevant_chunks)

print(f"✅ En benzer {top_k} bölüm bulundu")
print(f"📈 Benzerlik skorları: {similarities[most_similar_indices]}")

############################################
################ 10- GEMINI ################
############################################

print("Gemini'ye gönderiliyor...")

prompt_template = PromptTemplate.from_template("""

Sen CHP (Cumhuriyet Halk Partisi) hakkında bilgi veren bir asistansın.

Aşağıdaki CHP Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. 
Daha fazla bilgi için https://chp.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ Düzeltildi
    temperature=0,
    google_api_key=GEMINI_API_KEY  # ✅ Düzeltildi
)

############################################
################ 11- CHAIN #################
############################################

chain = prompt_template | llm | StrOutputParser()

############################################
################# 12- RUN ##################
############################################
response = chain.invoke({"context": context, "question": question})
print("\n" + "="*60)
print("Cevap:")
print("="*60)
print(response)
print("\n")
