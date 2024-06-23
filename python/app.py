import json
import boto3
import uvicorn
from fastapi import FastAPI

from s3_util import download_files_from_s3
from pdf_util import load_pdf_files_from_directory

app = FastAPI()

session = boto3.Session(profile_name="default")


@app.get("/")
async def root():
    return {"message": "Welcome to Contract Lens"}


@app.get("/predict/")
async def predict(prompt=None):
    runtime = session.client("runtime.sagemaker")
    llm_endpoint = "mistral-llm"
    if prompt is not None:
        response = runtime.invoke_endpoint(EndpointName=llm_endpoint, ContentType="application/json",
                                           Body=json.dumps({"inputs": f"<s>{prompt}[/INST]"}).encode("utf-8"))
        return json.loads(response['Body'].read().decode('utf-8'))
    else:
        return "No prompt provided"


if __name__ == "__main__":
    download_files_from_s3(session)
    text = load_pdf_files_from_directory("contracts")
    uvicorn.run(app, host="127.0.0.1", port=8000)
