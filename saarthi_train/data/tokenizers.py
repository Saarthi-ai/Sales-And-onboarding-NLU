import os
import json
import torch


class Tokenizer:
    """A custom tokenizer class written for the student model.

    Args:
        pretrained_tokenizer: Teacher model's pretrained tokenizer object.
        special_tokens (dict, optional): Dictionary containing the special tokens used. Defaults to None.
    """    
    def __init__(self, pretrained_tokenizer, special_tokens=None):
        self.vocab = None
        # self.subword_marker = subword_marker
        self.pretrained_tokenizer = pretrained_tokenizer

        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = self.pretrained_tokenizer.special_tokens_map

    def __call__(self, texts, padding, truncation, max_length, return_tensors=None, return_offsets_mapping=False):
        """Runs the entire tokenization pipeline on the given list of texts.

        Args:
            input (List[str]): List containing texts that need to be tokenized and encoded.
            padding (bool/str): Whether to pad the input or not. Can be True, False, or 'max_length'.
            truncation (bool): Whether sequences need to be truncated or not.
            max_length (int): Maximum sequence length.
            return_tensors (bool/str): Can be used to return tensors. Currently supported: 'pt' - PyTorch.
            return_offsets_mapping (bool): Can be used to return offsets per subword token that indicate the position of each subword in the original string.
        """
        pt_encoding = self.pretrained_tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_offsets_mapping=return_offsets_mapping
        )

        if type(texts) == str:
            out = self.pretrained_tokenizer.convert_ids_to_tokens(pt_encoding['input_ids'])
            out = self.convert_tokens_to_ids(out)
        else:
            out = list(map(self.pretrained_tokenizer.convert_ids_to_tokens, pt_encoding['input_ids']))
            out = list(map(self.convert_tokens_to_ids, out))

        attention_mask = pt_encoding['attention_mask']

        if return_tensors == 'pt':
            out = torch.tensor(out, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)

        result = {'input_ids': out, 'attention_mask': attention_mask}
        
        if return_offsets_mapping:
            result['offset_mapping'] = pt_encoding['offset_mapping']

        return result

    def fit(self, tokenized_texts):
        """Creates a vocabulary based on the tokenized texts provided to it.

        Args:
            tokenized_texts (List[List[str]]): List of tokenized texts.
        """
        all_tokens = []
        for tokenized_text in tokenized_texts:
            all_tokens += tokenized_text
        all_tokens = list(set(list(all_tokens)))

        all_tokens.insert(0, self.special_tokens['unk_token'])
        all_tokens.insert(0, self.special_tokens['eos_token'])
        all_tokens.insert(0, self.special_tokens['pad_token'])
        all_tokens.insert(0, self.special_tokens['bos_token'])

        self.vocab = {token: index for index, token in enumerate(all_tokens)}
        self.id2token = {index: token for token, index in self.vocab.items()}

    def convert_tokens_to_ids(self, tokens):
        """Converts a list of tokens to their ids.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[int]: List containing token ids.
        """        
        return [self.vocab.get(token, self.vocab[self.special_tokens['unk_token']]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Converts a list of tokens to their ids.

        Args:
            ids (List[int]): List of ids.

        Returns:
            List[str]: List containing tokens.
        """
        return [self.id2token[id] for id in ids]

    def get_vocab(self):
        """Return vocabulary

        Returns:
            dict: Vocabulary of the tokenizer.
        """        
        return self.vocab
    
    def get_special_tokens(self):
        """Returns special tokens, i.e. unk, eos, bos, pad.

        Returns:
            dict: Dictionary containing mapping of special tokens.
        """        
        return self.special_tokens
    
    def get_pad_index(self):
        return self.vocab[self.special_tokens['pad_token']]
    
    def vocab_size(self):
        """Returns size of vocabulary.

        Returns:
            int: Length of vocabulary dictionary.
        """        
        return len(self.vocab)
    
    def save_vocabulary(self, filepath, filename):
        """Saves tokenizer vocabulary as a json file.

        Args:
            filepath (str): Path where the vocabulary should be saved.
        """        
        with open(os.path.join(filepath, filename), 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, vocab):
        """Loads a dictionary into the vocabulary.

        Args:
            vocab (dict): Saved tokenizer vocabulary.
        """        
        self.vocab = vocab
        self.id2token = {v:k for k, v in self.vocab.items()}
    
    # def tokenize(self, text):
    #     """Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first 
    #     algorithm to perform tokenization using the given vocabulary.

    #     Args:
    #         text (str): Text to tokenize.
    #     """
    #     tokenized_result = []
    #     for token in text.split(' '):
    #         start = 0
    #         end = len(token)
    #         is_bad = False
    #         subtokens = []

    #         while start < len(token):
    #             longest_substr = ''
    #             substr = token[start:end] if start == 0 else self.subword_marker + token[start:end]
    #             if substr in self.vocab:
    #                 longest_substr = substr
    #                 break
    #             end = end - 1

    #         if longest_substr == '':
    #             is_bad = True
    #             break

    #         subtokens.append(longest_substr)
    #         start = end

    #         if is_bad:
    #             tokenized_result.append(self.vocab[self.special_tokens['unk']])
    #         else:
    #             tokenized_result.extend(subtokens)
        
    #     return tokenized_result

    # def detokenize(self, tokenized_text):
    #     """Detokenizes a list of tokens and converts it back into a string (joins on whitespace).

    #     Args:
    #         tokenized_text (List[str]): List of tokens.

    #     Returns:
    #         str: Detokenized string.
    #     """        
    #     detokenized_list = []
    #     token_builder = []

    #     for token in tokenized_text:
    #         if token.startswith(self.subword_marker):
    #             token_builder = list(map(lambda x: x.replace(self.subword_marker, ''), token_builder))
    #             detokenized_list.append(''.join(token_builder))

    #             token_builder.clear()
    #         token_builder.append(token)
    #     token_builder = list(map(lambda x: x.replace(self.subword_marker, ''), token_builder))
    #     detokenized_list.append(''.join(token_builder))

    #     return ' '.join(detokenized_list[1:])

