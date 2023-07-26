import uvicorn

import tensorflow as tf
import os
import numpy as np
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from fastapi import Request, FastAPI, Response
from fastapi.responses import JSONResponse
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification


model = AutoModelForCausalLM.from_pretrained("Ozgur98/pushed_model_mosaic_small", trust_remote_code=True).to(device='cuda:0', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

app = FastAPI(title="LLM Model")

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE, 
          response_model,
          response_model_exclude_unset=True)
async def predict(request: Request):
    body = await request.json()
    print(body)

    instances = body["instances"]
    print(instances)
    print(type(instances))
    instances = [x['text'] for x in instances]
    print(instances)

    tokenized_example = tokenizer(data, return_tensors='pt')
    outputs = model.generate(tokenized_example['input_ids'].to('cuda:0'), max_new_tokens=100, do_sample=True, top_k=10, top_p = 0.95)

    # Postprocess
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    prompt = answer[0].rstrip()
    
    return prompt
  
    

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)
