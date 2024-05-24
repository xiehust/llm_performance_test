from fastapi import FastAPI, Request
import uvicorn, json, datetime
# import torch
import boto3
import os
from transformers import AutoTokenizer
import sagemaker
from sagemaker import Model, image_uris, serializers, deserializers
import argparse

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

tokenizer,predictor = None, None

app = FastAPI()


@app.get("/ping")
def ping():
    return {'status': 'Healthy'}

@app.post("/generate")
async def create_item(request: Request):
    global tokenizer
    if not tokenizer:
        load_model()
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    print('json_post:',json_post)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('inputs') 
    parameters = json_post_list.get('parameters')
    max_new_tokens = parameters.get('max_new_tokens') if parameters.get('max_new_tokens') else 1000
    temperature = parameters.get('temperature') if parameters.get('temperature') else 0.5
    messages = [{"role":"user","content":prompt}]
    output = chat(messages=messages,max_new_tokens=max_new_tokens,temperature=temperature)
    answer = {
        "outputs": output,
        "status": 200
    }
    return answer


def chat(messages,max_new_tokens,temperature):
    global tokenizer,predictor
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f'input tokens:{len(inputs)}')
    parameters = {
            "max_new_tokens":max_new_tokens, 
            "do_sample": True,
            "stop_token_ids":[151645,151643],
            "repetition_penalty": 1.05,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 250
        }
    response = predictor.predict(
        {"inputs": inputs, "parameters": parameters}
    )
    return response['generated_text']

    
    
def load_model():
    global tokenizer,predictor

    MODEL_DIR = "Qwen/Qwen1.5-72B-Chat-AWQ"
    endpoint_name = 'lmi-model-qwen1-5-72B-2024-05-23-09-10-23-101'
    # role = sagemaker.get_execution_role()  # execution role for the endpoint
    sess = sagemaker.session.Session(boto3.session.Session())  # sagemaker session for interacting with different AWS APIs
    # region = sess._region_name  # region name of the current SageMaker Studio environment
    # account_id = sess.account_id()  # account_id of the current SageMaker Studio environment

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    predictor = sagemaker.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sess,
        serializer=serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer(),
    )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Chat RESTful API server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")
    parser.add_argument("--workers", type=int, default=10, help="current workers")
    
    args = parser.parse_args()
    
    load_model()
    print('model initialized')
    uvicorn.run("api:app", host=args.host, port=args.port,workers=args.workers)