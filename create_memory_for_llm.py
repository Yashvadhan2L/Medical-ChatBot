from langchain_community.document_loaders import PyPDFLoader ,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
import time 

# step 2 : load raw pdf 
DATA_PATH = 'C:\\Users\\yashs\\Desktop\\langchain\\working_cod\\data'
def load_pdf_files(data):
    loader = DirectoryLoader(data,glob= '*.pdf',loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
# print(documents)

print("length of PDF pages:", len(documents))

# step 2 : Create Chunks.. 
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks 

text_chunks  = create_chunks(documents)
# print(text_chunks)
# print(len(text_chunks))

# step 3 : Create vector Embeddings 
    # embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformer/all-Mini_L8-v2")
    # embedding_model = OllamaEmbeddings(model =' nomic-embed-text')


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="local_model",   #local_model --> sentence-transformers/all-MiniLM-L6-v2
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    return embedding_model


embedding_model = get_embedding_model()


# step 4 : Store embeddings in FAISS..
DB_FAIS_PATH = 'vectorstore/db_faiss'
db = FAISS.from_documents(text_chunks,embedding_model)
start = time.time()
db.save_local(DB_FAISS_PATH)
total_time = time.time() - start
print("Embeddings are saved... ")
print('Total_time : ', total_time)