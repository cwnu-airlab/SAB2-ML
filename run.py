import os
import sys
import time
import tarfile

import random
import numpy as np
import torch

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

os.environ['HYDRA_FULL_ERROR'] = '1'

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir

@hydra.main(version_base='1.2', config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    logging.info('COMMAND: python '+' '.join(sys.argv))

    if type(config.seed) != int:
        config.seed = round(time.time()) 
    logging.info(f'SET seed {config.seed}')
    set_seed(config.seed)

    config.job_num = '' if config.job_num == None else str(config.job_num)

    logging.info('<CONFIG>\n'+'\n'.join(print_config(config)))

    '''
    with open_dict(config):
        config.hydra = hydra.core.hydra_config.HydraConfig.get()
    '''

    if not config.checkpoint_path:
        config.checkpoint_path = os.path.join(config.save_dir, config.save_path)

    ## check checkpoint already is
    filelist = os.listdir(config.checkpoint_path)
    filelist = [d for d in filelist if 'log' not in d]
    filelist = [d for d in filelist if '.hydra' not in d]
    filelist = [d for d in filelist if '.yaml' not in d]
    filelist = [d for d in filelist if '.tar.gz' not in d]
    if config.mode in ['train'] and len(filelist) > 0:
        user_confirmation(f'"{config.checkpoint_path}" already havs {filelist}. Do you want overwrite? (y/n)')

    agent = hydra.utils.get_class(config.agent._target_)
    agent = agent(**config)
    save_code(config.checkpoint_path, mode=config.mode)

    agent.run()

    return None 

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.set_num_threads(1)
    #torch.set_num_interop_threads(1)

def save_code(path, mode='train'):
    codelist = [d.replace('.','/')+'.py' for d in sys.modules.keys() if 'src.' in d]
    codelist = [d for d in codelist if os.path.isfile(d)]
    codelist.append( os.path.join(path,'.hydra/') )

    with tarfile.open(os.path.join(path, f'code_{mode}.tar.gz'),'w:gz') as f:
        for filename in codelist:
            f.add(filename)
    
    ## save_conifg
    filename = os.path.join(path, '.hydra', 'config.yaml')
    with open(filename) as f:
        content = f.read()
    with open(os.path.join(path, f'config_{mode}.yaml'), 'w') as f:
        f.write(content+'\n')

def print_config(config_dict, level=0):
    if type(config_dict) != dict:
        config_dict = dict(config_dict)
    result = list()
    for key in config_dict:
        if type(config_dict[key]) == DictConfig:
            result.append(f"{'    '*level}[ {key} ]:\t(dict)")
            result += print_config(config_dict[key], level=level+1)
        else:
            result.append(f"{'    '*level}[ {key} ]:\t({type(config_dict[key]).__name__})\t{config_dict[key]}")
    return result

def user_confirmation(text):
    logging.warning(text)
    while(1):
        answer = input(">> ")
        if answer.lower() in ['n','no']: exit()
        elif answer.lower() in ['y','yes']: break
        else: continue

if __name__ == "__main__":
    main()
