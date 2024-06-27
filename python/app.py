import json
import boto3
import uvicorn
from fastapi import FastAPI, UploadFile, File
import os
from loader import load_documents
from llm import get_conversation_chain, get_llm_chain
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
session = boto3.Session(profile_name="default")
llm = None
vector = None


@app.get("/")
async def root():
    return {"message": "Welcome to Contract Lens"}


@app.get("/search/")
async def search(prompt=None, region=None, sector=None):
    search_kwargs = {'k': 4}
    if region is not None and sector is None:
        search_kwargs = {'filter': {'region': region}}
    elif region is None and sector is not None:
        search_kwargs = {'filter': {'sector': sector}}
    else:
        search_kwargs = {'filter': {'sector': sector, 'region': region}}

    vector_store_retriever = vector.as_retriever(search_kwargs=search_kwargs)
    if prompt is not None:
        docs = vector_store_retriever.invoke(prompt)
        response = []
        for doc in docs:
            data = {'id': doc.metadata['id'], 'content': doc.page_content, 'source': doc.metadata["source"],
                    'sector': doc.metadata["sector"], 'region': doc.metadata["region"]}
            response.append(data)
        return json.dumps(response)
    else:
        return "Please provide a prompt"


@app.get("/compare/")
async def compare(files: List[UploadFile] = File(...)):

    docs = []
    for file in files:
        print("loading file - "+ file.filename)
        docs.extend(PyPDFLoader(file).load())

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents=splits, embedding=embedding)

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_text"]


    content_handler = ContentHandler()

    runtime = boto3.client("runtime.sagemaker")

    llm=SagemakerEndpoint(
        endpoint_name="mistral-llm",
        client=runtime,
        model_kwargs={"temperature": 0.5, "max_new_tokens":750},
        content_handler=content_handler
    )

    template = """
    You are master in comparing two PDFs. 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in provided context just say, "Answer is not available in the context.", please don't provide the wrong answer.

    Provide Confidence Score as well.

    Context:\n {context}?\n
    Question: \nCompare and list out similarities and differences of contracts based on {question}.\n

    Confidence Score:
    """

    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = vector_db.as_retriever(search_kwargs={"k":12}),
        return_source_documents=False,
        chain_type = "stuff",
        chain_type_kwargs={"prompt":prompt}
    )


    question = "record dates"
    response = chain({"query":question})

    response_string = response.get("result")


    sub = "Question:"
    output = output + response_string.rsplit(sub, 1)[1]

    return output


@app.get("/chat/")
async def chat(prompt=None, id=None):
    if prompt is not None:
        llm_chain = get_llm_chain(llm, vector, id)
        answer = llm_chain({"query": prompt})
        return answer['result'].split('[/INST]')[0]
    else:
        return "How can I help you?"


@app.get("/documents/")
async def get_documents():
    path = "contracts"
    dir_list = os.listdir(path)
    return json.dumps(dir_list)


if __name__ == "__main__":
    vector = load_documents(session)
    llm = get_conversation_chain(session, vector)
    uvicorn.run(app, host="127.0.0.1", port=8000)
