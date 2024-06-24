from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from s3_util import download_files_from_s3
from pdf_util import load_pdf_files_from_directory


def load_documents(session):
    download_files_from_s3(session)
    text = load_pdf_files_from_directory("contracts")
    text_chunks = get_text_chunks(text)
    vector_db = get_vector_store(text_chunks)
    return vector_db


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1048,
        chunk_overlap=256,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(text_chunks, embedding)
    return vector_db
