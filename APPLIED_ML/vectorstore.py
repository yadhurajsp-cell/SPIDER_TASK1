from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

pdf_files = [
    "bert.pdf",
    "llama.pdf",
    "lm-fsl.pdf",
    "lora.pdf",
    "1706.03762v7.pdf"
   
]

documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)    
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma.from_documents(chunks, embedding=embeddings , persist_directory="chroma_db")

