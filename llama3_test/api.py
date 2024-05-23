from fastapi import FastAPI, Request
import uvicorn, json, datetime
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

tokenizer,model,terminators = None,None,None

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()


@app.get("/ping")
def ping():
    return {'status': 'Healthy'}

@app.post("/generate")
async def create_item(request: Request):
    global tokenizer,model,terminators
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
    torch_gc()
    return answer


def chat(messages,max_new_tokens,temperature):
    global tokenizer,model,terminators

    input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
                ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
    
def load_model():
    global tokenizer,model,terminators
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id,token='')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=''
    )
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]


    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
    
if __name__ == '__main__':
    load_model()
    print('model initialized')
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)