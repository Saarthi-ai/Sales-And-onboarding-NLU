from torch.profiler import profile, record_function, ProfilerActivity

from .data.preprocessing import remove_punctuations_and_convert_to_lowercase
from .inference.model_builder import get_finetuned_teacher, get_finetuned_student


model, tokenizer, max_len = get_finetuned_student('model')
text = 'this is a profiler specific input'

inputs = tokenizer([remove_punctuations_and_convert_to_lowercase(text)], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        out = model(**inputs)
        model.get_postprocessed_preds(out)

print(prof.key_averages().table(sort_by='cpu_time_total'))