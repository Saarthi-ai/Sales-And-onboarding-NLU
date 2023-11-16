import time

from .infer import init, run

from fastapi import FastAPI
from pydantic import BaseModel

import logging

start = time.time()
app = FastAPI()
init('model', 'student')
end = time.time() - start
print(f'Initialization took {end:.3f} seconds')

class ClassifierRequest(BaseModel):
    text: str
    lang: str = 'hindi'
    session_id: str = ''
    date: str | None = None
    consider_word_cnt: int = 0
    consider_weekend: bool = False
    consider_due_date: bool = True
    back_date: bool = False


@app.get('/')
async def health_check():
    return {'status': 'ok'}


@app.post("/predict")
async def predict(request: ClassifierRequest):
    try:
        return run(request.text, request.lang, request=dict(request))
    except Exception as e:
        logging.exception(f"Error during inference: {e}")
