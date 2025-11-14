############################################
################ 1- IMPORT #################
############################################

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # ✅ GEMINI YERİNE OLLAMA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")

############################################
############# 2- READ TO PDF ###############
############################################

print("PDF yükleniyor...")
loader = PyPDFLoader("data/chp.pdf")
pages = loader.load()
print(f"{len(pages)} sayfa yüklendi")

############################################
################ 3- CHUNKING ###############
############################################

print("Metin chunk'lara bölünüyor...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len
)

chunks = text_splitter.split_documents(pages)
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
####### 5- CREATE VECTOR DATABASE ##########
############################################

print("Vector database oluşturuluyor...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector database hazır")

############################################
########### 6- ASKING QUESTION #############
############################################

print("\n" + "="*60)
print("CHP Parti Tüzüğü - Soru-Cevap Sistemi (LOKAL QWEN)")
print("="*60)
question = input("\nSorunuz: ")
print(f"Aranıyor: '{question}'")

############################################
######### 7- SIMILARITY SEARCH #############
############################################

print("Benzerlik hesaplanıyor...")

top_k = 3
relevant_docs = vectorstore.similarity_search_with_score(question, k=top_k)

relevant_chunks = [doc.page_content for doc, score in relevant_docs]
context = "\n\n".join(relevant_chunks)

scores = [score for doc, score in relevant_docs]
print(f"En benzer {top_k} bölüm bulundu")
print(f"Benzerlik skorları: {scores}")

############################################
############ 8- LOKAL QWEN 2.5 #############
############################################

print("Lokal Qwen modeline gönderiliyor...")

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

# ✅ LOKAL QWEN MODEL
llm = Ollama(
    model="qwen2.5:7b-instruct-q4_K_M",
    temperature=0
)

############################################
################ 9- CHAIN ##################
############################################

chain = prompt_template | llm | StrOutputParser()

############################################
################ 10- RUN ###################
############################################

response = chain.invoke({"context": context, "question": question})
print("\n" + "="*60)
print("Cevap:")
print("="*60)
print(response)
print("\n")