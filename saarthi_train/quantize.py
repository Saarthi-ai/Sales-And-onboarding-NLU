import os
import json
import torch
from .inference.model_builder import build_teacher_from_checkpoint, build_student_from_checkpoint
from .optimizations.quantization import dynamic_quantization_for_transformers, dynamic_quantization_for_lstms
from .optimizations.torchscript import convert_lightning_transformer_classifier_to_ts, convert_lightning_lstm_classifier_to_ts


model_type = 'student'

if model_type == 'teacher':
    model_ckpt_path = os.path.join('model', model_type, 'lightning_logs', 'version_0', 'checkpoints', 'teacher_checkpoint.ckpt')
else:
    model_ckpt_path = os.path.join('model', model_type, 'lightning_logs', 'version_0', 'checkpoints', 'last.ckpt')
model_out_path = os.path.join('model', model_type, 'lightning_logs', 'version_0', 'checkpoints', 'model.pt')

print('Importing labels')
with open('./model/labels.json', 'r') as f:
    label_map = json.load(f)

print('Importing config')
with open('./model/training_config.json', 'r') as f:
    config = json.load(f)

print('Loading checkpoint')
checkpoint = torch.load(model_ckpt_path, map_location='cpu')

# change path for NLU and NER path
print('Initializing model')
model = eval(f"build_{model_type}_from_checkpoint('model')")
model.eval()

print('Quantizing model')
if model_type == 'teacher':
    model = dynamic_quantization_for_transformers(model, 'float16')
else:
    quantized = dynamic_quantization_for_lstms(model, 'int8')

print('Serializing to torchscript')
if model_type == 'teacher':
    convert_lightning_transformer_classifier_to_ts(model, config, model_out_path)
else:
    convert_lightning_lstm_classifier_to_ts(quantized, model_out_path) # replaced model with quantized

print('Finished!')
