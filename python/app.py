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
async def search(prompt=None):
    if prompt is not None:
        docs = vector.similarity_search_with_score(prompt)
        response = []
        for doc in docs:
            data = {'content': doc[0].page_content, 'source': doc[0].metadata["source"],
                    'sector': doc[0].metadata["sector"], 'region': doc[0].metadata["region"]}
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
async def chat(prompt=None, source=None):
    if prompt is not None and source is not None:
        llm_chain = get_llm_chain(llm, vector, source)
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
