import os
import re
import json
import pandas as pd
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataModuleBase(Dataset):
    def __init__(self, 
            tokenizer=None, 
            data_path=None, 
            **kwargs
        ):

        self.config = DictConfig(kwargs)
        self.tokenizer = tokenizer

        self.data = self.load_data(data_path)
        self.data_size = len(self.data)
        logging.info(f"LOAD {data_path}")

        max_length = 0
        if self.config.check_length and type(self.config.check_length) != str:
            for key in self.config.check_length:
                length_info = self.check_length([d[key] for d in self.data])
                logging.info(f'CHECK "{key}" Length:\n{length_info}')

        assert self.config.max_source_length > 0, \
                f'Need datamodule.config.max_source_length > 0 or datamodule.config.check_length is True. '+\
                f'But datamodule.config.max_source_length is {self.config.max_source_length} and datamodule.config.check_length is {self.config.check_length}'

    def load_data(self, filename):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def set_label_list(self):
        if self.config.label_file:
            if os.path.isfile(self.config.label_file):
                labels_path = self.config.label_file
            else:
                labels_path = os.path.join(self.config.data_dir, self.config.label_file)
            with open(labels_path,'r') as f:
                return [d.strip() for d in f]
        else:
            targets = [d['labels'] for d in self.data]
            return sorted(set(targets))

    def get_label_list(self):
        return self.label_list

    def check_length(self, data):
        length = list()
        for item in data:
            tokenized_source = self.tokenizer.tokenize(item)
            length.append(len(tokenized_source))
        return pd.Series(length).describe()
        
    def get_dataset(self):
        ## equal self class
        return self 

    def get_dataloader(self, sampler=None):
        dataloader = DataLoader(self,
                batch_size = self.config.batch_size, 
                shuffle = self.config.shuffle,
                num_workers = self.config.num_workers,
                sampler = sampler,
                collate_fn = lambda data: self.collate_fn(data))
        return dataloader

    def convert_sentence_to_input(self, inputs, max_len, direction='right', special_token=False):
        ## tokenizer.encode(text, max_length=max_length, padding='max_length', truncation=True)
        inputs = self.tokenizer.tokenize(inputs)
        return self.convert_tokens_to_input(inputs, max_len, direction=direction, special_token=special_token)
    
    def convert_tokens_to_input(self, inputs, max_len, direction='right', special_token=False):
        if special_token:
            inputs = [self.tokenizer.cls_token] + inputs + [self.tokenizer.sep_token] ## for bert

        dif = abs(max_len - len(inputs))
        if direction == 'left':
            if len(inputs) < max_len:  inputs = ( [self.tokenizer.pad_token] * dif ) + inputs
            elif max_len < len(inputs):  inputs = inputs[dif:]
        else:
            if len(inputs) < max_len:  inputs += [self.tokenizer.pad_token] * dif
            elif max_len < len(inputs):  inputs = inputs[:max_len]

        inputs = self.tokenizer.convert_tokens_to_ids(inputs)
        return inputs

    def convert_input_to_tokens(self, inputs, special_token=False):
        return self.tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=special_token)

    def convert_input_to_sentence(self, inputs, special_token=False):
        return self.tokenizer.decode(inputs, skip_special_tokens=special_token)

    def collate_fn(self, data):
        result = {
                'input_ids': [d['inputs'] for d in data],
                'labels': [d['labels'] for d in data],
                'data': [d['data'] for d in data]
                }

        for key in [d for d in result if d not in ['data']]:
            result[key] = torch.tensor(result[key])

        return result 



