import json
import boto3
import uvicorn
from fastapi import FastAPI

from loader import load_documents
from llm import get_conversation_chain

app = FastAPI()
session = boto3.Session(profile_name="default")
llm_chain = None


@app.get("/")
async def root():
    return {"message": "Welcome to Contract Lens"}


@app.get("/question/")
async def predict(prompt=None):
    answer = llm_chain({"query": prompt})
    return answer['result'].split('[/INST]')[0]


if __name__ == "__main__":
    llm_chain = get_conversation_chain(session, load_documents(session))
    uvicorn.run(app, host="127.0.0.1", port=8000)
