import os
import sys
import copy
import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import numpy as np
import torch
import transformers


from .sab import sab2_ml as sab2_ml

import math
import statistics
from itertools import combinations
from collections import Counter, defaultdict
import random


class Agent():
    def __init__(self,
            **kwargs
        ):
        self.config = DictConfig(kwargs)
        self.training_args = OmegaConf.to_container(self.config, resolve=True)

        model = hydra.utils.get_class(self.config.model._target_)

        ## Setting
        self.tokenizer = model.set_tokenizer(self.config.tokenizer) ## set tokenizer
        self.set_data(self.config.mode) ## set data and labels
        self.model = model(tokenizer=self.tokenizer, **self.config.model) ## set model
        self.optimizer = self.model.configure_optimizers(**self.config.optimizer)
                
    def run(self):
        if self.config.mode == 'train':
            self.fit()
        elif self.config.mode == 'predict':
            self.predict()
        else:
            raise NotImplementedError('OPTION "{}" is not supported'.format(self.config.mode))


    ######## SETTING #######################################################

    def set_dataloader(self, config, **kwargs):
        ## source file: src/datamodule/
        config = dict(config)
        config['data_path'] = kwargs['data_path']
        config['tokenizer'] = kwargs['tokenizer']

        datamodule = hydra.utils.instantiate(config)
        dataloader = datamodule.get_dataloader()
        return dataloader

    def set_data(self, mode):
        def set_dataloader(target_file):
            if os.path.isfile(target_file):
                filename = target_file
            else:
                filename = os.path.join(self.config.work_dir, self.config.datamodule.data_dir, target_file)
            return self.set_dataloader(self.config.datamodule, tokenizer=self.tokenizer, data_path=filename)

        if mode in ['train']:
            self.train_dataloader = set_dataloader(self.config.datamodule.train_data)
            self.valid_dataloader = set_dataloader(self.config.datamodule.valid_data)
            if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
                self.test_dataloader = set_dataloader(self.config.datamodule.test_data)

        elif mode in ['predict']:
            self.test_dataloader = set_dataloader(self.config.datamodule.test_data)

    ########################################################################



    def gini(self, category_sizes) :
        C = len(category_sizes)                
        n = np.sum(category_sizes)
        absolute_differences = np.sum(np.abs(category_sizes[:, None] - category_sizes))
        gini = absolute_differences / (2 * n * C)
        return gini
    
    
    def fit(self): ## TRAINING

        earlystop_threshold = self.config.agent.patience
        patience = 0
        max_loss = 999
        self.best_model = None

        before_gradient = 0.0
        now_gradient = 0.0
        loss_before = 0.0
        loss_now = 0.0
        warm_up = self.config.agent.sampling_warmup
        is_update = False

        def should_start_sampling(val_loss):
            nonlocal loss_before, loss_now, now_gradient, before_gradient
            if loss_before == 0.0:
                loss_before = loss_now = val_loss
                return False

            loss_before, loss_now = loss_now, val_loss
            diff = loss_before - loss_now

            if now_gradient == 0.0:
                now_gradient = diff
                return False

            before_gradient, now_gradient = now_gradient, diff
            if now_gradient < before_gradient:
                return True

            logging.info('continue warm up...')
            return False


        after_threshold = self.config.agent.after_patience
        after_patience = 0

        total_label = [data['labels'] for data in self.train_dataloader.dataset.data]
        class_counts = Counter(label for sample in total_label for label in sample)

        total_sum = sum(class_counts.values())
        weight_dict = {cls: total_sum / count for cls, count in class_counts.items()}

        class_data = dict(sorted(class_counts.items(), key=lambda x: x[0]))
        weight = np.fromiter(class_data.values(), dtype=float)

        over_weight = np.array(
            [np.mean([weight_dict[label] for label in sample]) for sample in total_label],
            dtype=float
        )
        over_weight /= over_weight.sum()

        coefficient = self.gini(weight)

        self.sampler = sab2_ml.Sampler(
                self.train_dataloader.dataset.data_size, 
                len(self.train_dataloader.dataset.label_list), 
                self.config.agent.sampling_warmup, 
                [100.0,1.0], 
                [self.config.agent.sampling_warmup, self.config.agent.epochs],
                total_label) 
        def set_sampler(epoch=None):
            if epoch is not None:
                logging.info('Update sampler')
                self.sampler.update_sampling_probability(normalize=self.config.agent.norm)
            self.model.eval()
            logits, ids = list(),list()
            self.train_dataloader = self.train_dataloader.dataset.get_dataloader()
            for bindex, batch in enumerate(tqdm(self.train_dataloader, desc='[SET sampler prob]')):
                output = self.model.validation_step(batch)
                ids += batch['index']
                #logits 는 [[a,b],[c,d]] 상태를 펼처서 [a,b,c,d]로 만든다.
                #logits += output['logits'].view(-1).cpu().tolist()
                logits += output['logits'].cpu().tolist()


            self.sampler.async_update_prediction_matrix(ids, logits)
            prob_table = self.sampler.prob_table.table
            return prob_table

        for epoch in range(self.config.agent.epochs):
            print('',flush=True)

            if epoch < warm_up:
                set_sampler()
            else: 
                sab_weight = set_sampler(epoch=epoch)
                total_prob = (1- coefficient) * sab_weight + coefficient * over_weight
                print(total_prob)
                sampler = torch.utils.data.WeightedRandomSampler(total_prob, self.train_dataloader.dataset.data_size)
                if after_patience == 1 :
                    logging.info('Sequential Sampling after early stop')
                    self.train_dataloader = self.train_dataloader.dataset.get_dataloader()
                elif after_patience !=0 and patience != earlystop_threshold :
                    logging.info('Sequential Sampling')
                else :
                    logging.info('SAB2-ML sampling')
                    self.train_dataloader = self.train_dataloader.dataset.get_dataloader(sampler=sampler)
            
            ## training step
            self.model.train()
            tr_loss = tr_acc = 0
            dataloader = tqdm(self.train_dataloader)#, ascii=True)
            batch_storage = list()
            for index, batch in enumerate(dataloader): ## 1 epoch
                batch_storage.append(batch['data'])
                output = self.model.training_step(batch)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                tr_loss += loss.item()
                tr_acc += acc.item()

                process_description = f"[TRAIN]_Epoch{epoch}_L{tr_loss/(index+1):.3f}_A{tr_acc/(index+1):.3f}_time{dataloader.format_dict['elapsed']:.4f}_"
                logging.debug(process_description+f'step{index}_')
                dataloader.set_description(process_description)
            logging.info(process_description)

            ## validation step
            self.model.eval()
            val_loss = val_acc = 0
            dataloader = tqdm(self.valid_dataloader)#, ascii=True)
            for index, batch in enumerate(dataloader): ## 1 epoch
                output = self.model.validation_step(batch)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                val_loss += loss.item()
                val_acc += acc.item()

                process_description = f"[VALID]_Epoch{epoch}_L{val_loss/(index+1):.3f}_A{val_acc/(index+1):.3f}_time{dataloader.format_dict['elapsed']:.4f}_"
                logging.debug(process_description+f'step{index}_')
                dataloader.set_description(process_description)
            logging.info(process_description)

            if self.config.agent.model_all_save:
                path = os.path.join(
                        self.config.checkpoint_path,
                        f"train_{self.config.model.name}" +\
                        f"_lr_{self.config.optimizer.lr:.0E}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss/(index+1):.4f}" +\
                        f"_acc_{val_acc/(index+1):.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)

            should_check = (not is_update) and epoch < self.config.agent.sampling_warmup
            if should_check and should_start_sampling(val_loss):
                logging.info('Start sampling')
                is_update = True
                warm_up = epoch + 1
                self.sampler.update_queue_epoch_size(warm_up)

            ## earlystop
            if epoch < warm_up : continue
            val_loss = val_loss/(index+1)
            val_acc = val_acc/(index+1)
            if val_loss < max_loss and after_patience == 0 :
                max_loss = val_loss
                patience = 0
                path = os.path.join(
                        self.config.checkpoint_path,
                        f"valid_{self.config.model.name}" +\
                        f"_lr_{self.config.optimizer.lr:.0E}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss:.4f}"+\
                        f"_acc_{val_acc:.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)
                path = os.path.join( self.config.checkpoint_path, "trained_model" )
                self.model.save_model(path)
                self.best_model = copy.deepcopy(self.model)
            elif val_loss < max_loss and after_patience != 0:
                logging.info('update_loss but already early Stop')
                after_patience +=1

                max_loss = val_loss
                path = os.path.join( self.config.checkpoint_path, "trained_model_after_training" )
                self.model.save_model(path)
                self.after_model = copy.deepcopy(self.model)
                if after_patience > after_threshold :
                    logging.info('Ran out of patience')
                    break

            else:
                patience += 1
                if patience > earlystop_threshold : #patience is earlystop_threshold + 1
                    after_patience += 1
                    if after_patience > after_threshold :
                        path = os.path.join(self.config.checkpoint_path, 'trained_model_after_training')
                        self.model.save_model(path)
                        self.after_model = copy.deepcopy(self.model)
                        logging.info('Ran out of patience')
                        break

                    #break ## STOP training
        ## predict best model
        if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
            path = os.path.join(self.config.checkpoint_path, 'trained_model_after_training')
            self.model = copy.deepcopy(self.after_model)
            self.predict(dir_path=path)


    
    def predict(self, dir_path=None):
        self.model.eval()
        if not dir_path:
            dir_path = self.config.model.path

        if self.config.mode == 'train':
            if not self.config.predict_file_path:
                raise ValueError('No empty predict_file_path supported. Make sure predict_file_path has a value.')
            ofp_name = os.path.join( dir_path, self.config.predict_file_path )
            ofp = open(ofp_name,'w')
            logging.info(f"WRITE {ofp_name}")
        elif self.config.predict_file_path:
            ofp_name = os.path.join( dir_path, self.config.predict_file_path )
            ofp = open(ofp_name,'w')
            logging.info(f"WRITE {ofp_name}")
        else:
            ofp = sys.stdout
            logging.info(f"WRITE sys.stdout")

        dataloader = tqdm(self.test_dataloader)
        for index, batch in enumerate(dataloader): ## 1 epoch
            output = self.model.predict_step(batch)

            for out in output:
                ofp.write(json.dumps(out, ensure_ascii=False)+'\n')
            dataloader.set_description(f"[PREDICT]")

        config = OmegaConf.to_container(self.config, resolve=True)
        ofp.write(json.dumps(config, ensure_ascii=False)+'\n')
        ofp.close()
