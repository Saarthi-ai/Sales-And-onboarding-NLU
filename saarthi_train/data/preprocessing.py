import re
import sys
import torch
import multiprocessing as mp


def preprocess_df(input_df, label_map, max_seq_len, tokenizer):
    """Preprocesses a dataframe for the text classification task.

    Args:
        input_df (pandas.DataFrame): DataFrame object containing the raw training data.
        label_map (dict): Dictionary containing lists of all the output labels.
        max_seq_len (int): Maximum sequence length of the model.
        tokenizer: Model's tokenizer.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """    
    result = input_df.copy(deep=True)
    for col in input_df:
        if col not in label_map.keys():
            result[f'{col}_token_ids'] = preprocess_text(result[col].tolist(), max_seq_len, tokenizer)
            del result[col]

    for col in result:
        if col not in label_map.keys():
            continue
        result[col] = result[col].map(lambda x: encode_label(label_map, col, x))

    return result


def preprocess_sequence_df(input_df, task_name, label_map, max_seq_len, tokenizer):
    """Preprocesses a dataframe for the text classification task.

    Args:
        input_df (pandas.DataFrame): DataFrame object containing the raw training data.
        task_name (str): Name of the sequence classification task. Has to be the same as the name in the labels.json file. For ex - ner, pos, etc.
        label_map (dict): Dictionary containing lists of all the output labels.
        max_seq_len (int): Maximum sequence length of the model.
        tokenizer: Model's tokenizer.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """    
    result = input_df.copy(deep=True)

    result['text'] = result['text'].apply(remove_punctuations_and_convert_to_lowercase_ner)
    result['text_token_ids'], result['token_tags'] = zip(*result.apply(lambda row: preprocess_sequence_with_subword_alignment(row['text'], row[task_name], tokenizer, max_seq_len), axis=1))
    result[task_name] = result['token_tags'].map(lambda x: encode_sequence_tags(label_map, task_name, x))
    del result['text']
    del result['token_tags']

    return result


def preprocess_text(input, max_seq_len, tokenizer):
    """Preprocesses a single text.

    Args:
        input (str): A single input text to the model.
        max_seq_len (int): Maximum sequence length of the model.
        tokenizer: Model's tokenizer.

    Returns:
        torch.Tensor: Preprocessed and encoded text.
    """    
    if isinstance(input, str):
        filtered = remove_punctuations_and_convert_to_lowercase(input)
    elif isinstance(input, list):
        pool = mp.Pool()
        filtered = pool.map(remove_punctuations_and_convert_to_lowercase, input)

    tokenizer_out = tokenizer(
        filtered,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len,
        return_tensors='pt'
    )

    out_keys = list(tokenizer_out.keys())
    result = [{out_keys[0]: value.tolist()} for value in torch.unbind(tokenizer_out[out_keys[0]])]
    for key in out_keys[1:]:
        for idx, value in enumerate(torch.unbind(tokenizer_out[key])):
            result[idx][key] = value.tolist()

    return result


def preprocess_sequence_with_subword_alignment(text, tags, tokenizer, max_seq_len):
    """Helper function to preprocesses a text and align the tags with sub-words for sequence classification tasks.

    Args:
        text (str): A single input text to the model.
        tags (list): A string of tags for the corresponding input text.
        tokenizer: Model's tokenizer.
        max_seq_len (int): Maximum sequence length of the model.

    Returns:
        dict: Dictionary containing encoded text input.
        list: List of tags aligned in accordance with the sub-word tokenization for the corresponding input text.
    """    
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len
    )
    attention_mask = encoding['attention_mask']
    offset_mapping = encoding['offset_mapping']

    tag_list = tags.split()

    token_tags = []
    word_index = 0
    prev_end_position = 0
    prev_start_position = 0
    # print(f'Input text: {text}')
    # print(f'Tokenized text: {tokenizer.tokenize(text)}')
    # print(f'Attention Mask: {attention_mask}')
    # print(f'Offset Mapping: {offset_mapping}')
    # print(f"Tag list: {tag_list}")
    for i, offset in enumerate(offset_mapping):
        if attention_mask[i] == 0:
            token_tags.append('<pad>')
            continue

        if offset[0] == 0 and offset[1] == 0:
            token_tags.append('O')
            continue

        if offset[0] > prev_end_position:
            word_index += 1

        try:
            token_tags.append(tag_list[word_index])
        except IndexError:
            while len(tag_list) >= len(token_tags):
                try:
                    token_tags.append(tag_list[word_index-1])
                except IndexError:
                    print('ERROR WHILE PREPROCESSING TAGS:')
                    print(f'Input text: {text}')
                    print(f'Tokenized text: {tokenizer.tokenize(text)}')
                    sys.exit(f"Missing tag at index: {i}. Tag list: {tag_list}")

        prev_end_position = offset[1]
        # print(prev_end_position)
    try:
        assert(len(token_tags) == max_seq_len)
    except: 
        token_to_add = ['O']*(max_seq_len-len(token_tags))
        token_tags.extend(token_to_add)
    print(token_tags)
    
    return {key: value for key, value in encoding.items() if key != 'offset_mapping'}, token_tags


def remove_punctuations_and_convert_to_lowercase(text):
    """Removes special characters and converts string to lowercase.

    Args:
        text (str): Input string.

    Returns:
        str: Preprocessed input.
    """    
    punc_pattern = r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~।]'
    return ' '.join(re.sub(punc_pattern, ' ', text.lower()).split())


def remove_punctuations_and_convert_to_lowercase_ner(text):
    """Removes special characters converts string to lowercase for the NER task.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text with special characters removed.
    """    
    punc_pattern = r'[!"#$%&()*+;<=>?@[\]^_`{|}~।]'
    
    while re.search(r'(\d)\s*,\s*(\d)', text):
        text = re.sub(r'(\d)\s*,\s*(\d)', r'\1\2', text)
    return ' '.join(re.sub(punc_pattern, ' ', text.lower()).split())


def encode_label(label_map, output_name, label):
    """Encodes labels to numeric indices.

    Args:
        label_map (dict): Dictionary containing lists of all the output labels.
        output_name (str): Name of the model output that the label belongs to.
        label (str): Label to be encoded.

    Returns:
        int: Encoded label.
    """    
    # print(label, output_name)
    return label_map[output_name].index(label)


def encode_sequence_tags(label_map, task_name, tags):
    """Encodes sequence tags to numeric indices.

    Args:
        label_map (dict): Dictionary containing lists of all the output labels.
        task_name (str): Name of the sequence classification task. For ex - ner, pos, etc.
        tags (list): List of tags that need to be encoded.

    Returns:
        list: Encoded tags.
    """    
    return [label_map[task_name].index(tag) for tag in tags]
