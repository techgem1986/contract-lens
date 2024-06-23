import boto3
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Contract Lens"}


@app.get("/sagemaker_endpoint")
async def sagemaker_endpoint():
    session = boto3.Session(profile_name="techgem1986")
    sagemaker_client = session.client("sagemaker")
    response = sagemaker_client.list_endpoints()
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
