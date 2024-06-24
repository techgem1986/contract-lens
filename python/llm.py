from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Dict
import json

prompt_template = """
You are an AI chatbot. You can read documents and provide accurate information based on the source.
You are an expert in reading American Depositary Receipts contracts.
<<< f{context} >>>

#Instructions: ##Summarize: Please provide an accurate answer based on the context with reference of the context 
supporting your answer, if the answer cannot be generated from the context, then reply 'The answer cannot be 
generated from the given context'. Generate answer under 150 words. Respond with format Question:, Answer: , 
Reference: , Confidence Score

<<< Question: f{question} >>>
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps(
            {"inputs": f"<s>[INST] {prompt} [/INST]", "parameters": {**model_kwargs}}
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        splits = response_json[0]["generated_text"].split("[/INST] ")
        return splits[0]


model_kwargs = {"max_new_tokens": 1024, "temperature": 0.8, "do_sample": False}
content_handler = ContentHandler()


def get_conversation_chain(session, vector_db):
    sm_client = session.client("sagemaker-runtime")
    llm = SagemakerEndpoint(
        endpoint_name="mistral-llm",
        model_kwargs=model_kwargs,
        content_handler=content_handler,
        client=sm_client,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
