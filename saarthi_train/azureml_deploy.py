import os
import json

from .infer import init as initialize_model, run as run_model


def init():
    model_root = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs')
    initialize_model(model_root)


def run(data):
    data = json.loads(data)['data']
    return run_model(data)
