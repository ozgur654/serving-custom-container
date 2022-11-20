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

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForSequenceClassification.from_pretrained("../sentiment")

app = FastAPI(title="Sentiment Analysis")

AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

class Prediction(BaseModel):
  sentiment: str 
  confidence: Optional[float]

class Predictions(BaseModel):
    predictions: List[Prediction]

# instad of creating a class we could have also loaded this information
# from the model configuration. Better if you introduce new labels over time
class Sentiment(Enum):
  NEGATIVE = 0
  POSITIVE = 1


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE, 
          response_model=Predictions,
          response_model_exclude_unset=True)
async def predict(request: Request):
    body = await request.json()
    print(body)

    instances = body["instances"]
    print(instances)
    print(type(instances))
    instances = [x['text'] for x in instances]
    print(instances)

    tf_batch = tokenizer(instances, max_length=128, padding=True,
                            truncation=True, return_tensors='tf')

    print(tf_batch)

    tf_outputs = model(tf_batch)

    print(tf_outputs)

    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    print(tf_predictions)

    indices = np.argmax(tf_predictions, axis=-1)
    confidences = np.max(tf_predictions, axis=-1)

    outputs = []

    for index, confidence in zip(indices, confidences):
      sentiment = Sentiment(index).name
      print(index)
      print(confidence)
      outputs.append(Prediction(sentiment=sentiment, confidence=confidence))

    return Predictions(predictions=outputs)

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0",port=8080)