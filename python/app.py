import boto3
from fastapi import FastAPI
import uvicorn
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Contract Lens"}


@app.get("/predict/")
async def sagemaker_endpoint(prompt:str = None):
    session = boto3.Session(profile_name="default")
    runtime = session.client("runtime.sagemaker")
    llm_endpoint="mistral-llm"
    if prompt is not None:
        response = runtime.invoke_endpoint(EndpointName=llm_endpoint, ContentType="application/json", Body=json.dumps({"inputs":f"<s>{prompt}[/INST]"}).encode("utf-8"))
        return json.loads(response['Body'].read().decode('utf-8'))
    else:
        return "No prompt provided"



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
