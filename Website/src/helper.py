from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob = "*.pdf",
                    loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf("data/")
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap = 10)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def mycall():
   text_chunks = text_split(extracted_data)
   return text_chunks

#create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap = 10)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings