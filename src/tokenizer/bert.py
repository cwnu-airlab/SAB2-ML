import transformers
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

class Tokenizer():
    def __init__(self, 
            path,
            **kwargs,
            ):
        self.config = DictConfig(kwargs)

        self.tokenizer = transformers.BertTokenizer.from_pretrained(path, truncation_side=self.config.side, padding_side=self.config.side)
        logging.info(f'LOAD tokenizer: {path}')

    def get_tokenizer(self):
        symbols = '!#$%&\"\'()*+,-./0123456789;:<=>?@'
        hangle = '다람쥐 헌 쳇바퀴에 타고파'
        alphabet  = 'The Quick Brown Fox Jumps Over The Lazy Dog'
        for sample in [symbols, hangle, alphabet]:
            logging.info(f"\nTokenized sample:\n{sample}\n=> {self.tokenizer.tokenize(sample)}")
        return self.tokenizer
        
