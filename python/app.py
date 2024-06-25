import json
import boto3
import uvicorn
from fastapi import FastAPI, UploadFile, File
import os
from loader import load_documents
from llm import get_conversation_chain, get_llm_chain
from typing import List

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


@app.post("/compare/")
async def compare(files: List[UploadFile] = File(...)):
    for file in files:
        print(file.filename)
    return "Functionality to be implemented"


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
