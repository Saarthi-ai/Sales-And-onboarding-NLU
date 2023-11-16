FROM python:3.11.5-slim-bullseye

RUN mkdir nlu
COPY . /nlu

WORKDIR /nlu
RUN pip install -r nlu_requirements.txt --no-cache-dir

ENTRYPOINT ["uvicorn", "saarthi_train.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

