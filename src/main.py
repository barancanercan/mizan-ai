"""
Turkish Government Intelligence Hub - RAG System
Basit ve temiz fonksiyonlarla organize edilmiş
"""

############################################
################ 1- IMPORT #################
############################################

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")

############################################
############# 2- READ TO PDF ###############
############################################

def load_pdf(pdf_path):
    """PDF dosyasını yükle"""
    print("PDF yükleniyor...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"{len(pages)} sayfa yüklendi")
    return pages

############################################
################ 3- CHUNKING ###############
############################################

def chunk_documents(pages, chunk_size=512, chunk_overlap=50):
    """Dökümanları chunk'lara böl"""
    print("Metin chunk'lara bölünüyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    print(f"{len(chunks)} chunk oluşturuldu")
    return chunks

############################################
########## 4- LOAD EMBEDDING MODEL #########
############################################


def load_embeddings(model_name="nezahatkorkmaz/turkce-embedding-bge-m3"):
    """Türkçe embedding modelini yükle"""
    print("Türkçe Embedding Modeli yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding modeli hazır")
    return embeddings

############################################
####### 5- CREATE VECTOR DATABASE ##########
############################################

def create_vectorstore(chunks, embeddings, persist_dir="../chroma_db"):
    """Vector database oluştur"""
    print("Vector database oluşturuluyor...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("Vector database hazır")
    return vectorstore

############################################
######### 6- SIMILARITY SEARCH #############
############################################

def search_similar_docs(vectorstore, question, top_k=3):
    """Benzer dökümanları bul"""
    print(f"Aranıyor: '{question}'")
    print("Benzerlik hesaplanıyor...")
    
    relevant_docs = vectorstore.similarity_search_with_score(question, k=top_k)
    relevant_chunks = [doc.page_content for doc, score in relevant_docs]
    context = "\n\n".join(relevant_chunks)
    scores = [score for doc, score in relevant_docs]
    
    print(f"En benzer {top_k} bölüm bulundu")
    print(f"Benzerlik skorları: {scores}")
    
    return context

############################################
########## 7- SETUP LLM CHAIN ##############
############################################

def setup_llm_chain(model_name="qwen2.5:7b-instruct-q4_K_M", temperature=0):
    """LLM ve prompt chain'i hazırla"""
    print("Lokal Qwen modeli hazırlanıyor...")
    
    prompt_template = PromptTemplate.from_template("""
Sen CHP (Cumhuriyet Halk Partisi) hakkında bilgi veren bir asistansın.

Aşağıdaki CHP Parti Tüzüğü bölümüne göre soruyu yanıtla:

{context}

Kullanıcının Sorusu: {question}

Yanıt Kuralları:
- Kibar, nazik ve bilgilendirici ol
- Doğrudan cevap ver, kaynak belirtme
- Eğer ilgili bilgi yukardaki metinde yoksa: "Bu konuda parti tüzüğünde detaylı bilgi bulamadım. Daha fazla bilgi için https://chp.org.tr/ adresini ziyaret edebilirsiniz."

Yanıt:
""")
    
    llm = Ollama(model=model_name, temperature=temperature)
    chain = prompt_template | llm | StrOutputParser()
    
    return chain

############################################
########## 8- GENERATE ANSWER ##############
############################################

def generate_answer(chain, context, question):
    """LLM ile cevap üret"""
    response = chain.invoke({"context": context, "question": question})
    return response

############################################
############ 9- CREATE MAIN ################
############################################

def main():
    """Ana program - çoklu soru sorma özelliği ile"""
    
    # 1. PDF'i yükle ve hazırla
    pages = load_pdf("../data/chp.pdf")
    chunks = chunk_documents(pages)
    
    # 2. Embedding ve Vector DB
    embeddings = load_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    
    # 3. LLM Chain hazırla
    chain = setup_llm_chain()
    
    # 4. Soru-cevap döngüsü
    print("\n" + "="*60)
    print("CHP Parti Tüzüğü - Soru-Cevap Sistemi (LOKAL QWEN)")
    print("="*60)
    print("Çıkmak için 'q' veya 'quit' yazın\n")
    
    while True:
        question = input("\nSorunuz: ").strip()
        
        # Çıkış kontrolü
        if question.lower() in ['q', 'quit', 'exit', 'çıkış']:
            print("\nGörüşmek üzere! 👋")
            break
        
        if not question:
            print("Lütfen bir soru yazın.")
            continue
        
        # Cevap üret
        context = search_similar_docs(vectorstore, question)
        response = generate_answer(chain, context, question)
        
        print("\n" + "="*60)
        print("Cevap:")
        print("="*60)
        print(response)
        print("\n" + "="*60)

############################################
################ 10- RUN ###################
############################################

if __name__ == "__main__":
    main()
