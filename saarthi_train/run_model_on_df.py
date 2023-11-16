import time
import json
import pandas as pd
from tqdm import tqdm
from .infer import init, run


print('Initializing model')
init('./model', 'student')
df = pd.read_csv('../tamil_test_set.csv', engine='pyarrow', dtype_backend='pyarrow')
keys = [k for k in df if k != 'text']

preds = {f'{k}_preds': [] for k in keys}
confs = {f'{k}_confs': [] for k in keys}
correct = {f'{k}_correct?': [] for k in keys}

print('Running inference..')
start = time.time()
prog_bar = tqdm(df.iterrows(), total=len(df))
for idx, row in prog_bar:
    model_out = run(row['text'])

    for k in keys:
        preds[f'{k}_preds'].append(model_out[k][0]['name'])
        confs[f'{k}_confs'].append(model_out[k][0]['confidence'])
        correct[f'{k}_correct?'].append(row[k] == model_out[k][0]['name'] and model_out[k][0]['confidence'] >= 0.6)
end = time.time() - start
print(f'Inference done. Time taken: {end} seconds.')
results_df = pd.DataFrame({**preds, **confs, **correct})
final_df = pd.concat([df, results_df], axis=1)

print('Calculating accuracy..')
acc = 0
prog_bar = tqdm(final_df.iterrows(), total=len(final_df))
for idx, row in prog_bar:
    correct = True
    for k in keys:
        correct = correct and row[f'{k}_correct?']
    
    if correct:
        acc += 1

acc_dict = {k: sum(final_df[f'{k}_correct?']) / len(final_df) for k in keys}
total_acc = acc / len(final_df)

print(f'Overall accuracy: {total_acc}')
print(f'Accuracy per output:')
print(json.dumps(acc_dict, indent=2))
