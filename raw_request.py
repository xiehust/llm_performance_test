import requests
import json

text1 = \
"""你是一名小说家，热衷于创意写作和编写故事。 
请帮我编写一个故事，对象是10-12岁的小学生
故事背景：
讲述一位名叫莉拉的年轻女子发现自己有控制天气的能力。她住在一个小镇上，每个人都互相认识。
其他要求：
-避免暴力，色情，粗俗的语言
-长度要求不少于500字
请开始：
"""
text2 = """Please translate below text to Chinese:
Prompt engineering is a relatively new discipline for developing and optimizing prompts to efficiently use language models (LMs) for a wide variety of applications and research topics.
Prompt engineering skills help to better understand the capabilities and limitations of large language models (LLMs).Researchers use prompt engineering to improve the capacity of LLMs on a wide range of common and complex tasks such as question answering and arithmetic reasoning.
Developers use prompt engineering to design robust and effective prompting techniques that interface with LLMs and other tools.Prompt engineering is not just about designing and developing prompts. It encompasses a wide range of skills and techniques that are useful for interacting and developing with LLMs. It's an important skill to interface, build with, and understand capabilities of LLMs. You can use prompt engineering to improve safety of LLMs and build new capabilities like augmenting LLMs with domain knowledge and external tools.Motivated by the high interest in developing with LLMs, we have created this new prompt engineering guide that contains all the latest papers, learning guides, models, lectures, references, new LLM capabilities.
"""
text3 = "hello"

prompt_template = "<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"

# text = """tell me three fruit"""
# url = "http://34.229.46.119:8080/generate"
url = "http://3.229.113.93:8080/generate"

prompt = prompt_template.format(user_message=text1)
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 500,
        "temperature": 0.1,
        # "stop_tokens": 128009
    }
}
headers = {
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, json=payload, headers=headers)

# print(json.loads(response.text)['generated_text'])
print(response.text)