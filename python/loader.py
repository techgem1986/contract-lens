from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import PyPDF2
import glob
import os

from s3 import download_files_from_s3

s3_bucket_name = "contractlens"
s3_bucket_prefix = "contracts"


def load_documents(session):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    download_files_from_s3(session, s3_bucket_name, s3_bucket_prefix)
    list_of_documents = load_pdf_files_from_directory(s3_bucket_prefix)
    vector_db = get_vector_store(list_of_documents, embedding)
    return vector_db


def load_pdf_files_from_directory(directory):
    list_of_documents = []
    doc_id = 0
    for file in glob.glob(directory + "/*.pdf"):
        if file.endswith('.pdf'):
            doc_id = doc_id + 1
            doc_name = file.split(os.path.sep)[1]
            file_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(file_reader.pages):
                chunks = get_text_chunks(page.extract_text())
                for j, chunk in enumerate(chunks):
                    document = Document(
                        page_content=chunk,
                        metadata={"source": doc_name, "page": i, "sector": "Automobile",
                                  "region": "APAC", "genre": "Contract", "id": doc_id},
                    )
                    list_of_documents.append(document)
    return list_of_documents


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1048,
        chunk_overlap=256,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(list_of_documents, embedding):
    vector_db = FAISS.from_documents(list_of_documents, embedding)
    vector_db.save_local("faiss_index")
    vector_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    return vector_db
