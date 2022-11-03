# -*- encoding: utf-8 -*-

import re
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import json
import random
import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import torch.optim as optim

#===============================================Data=====================================================
class DataManager:
    def __init__(self):
        
        dp = '../bertlarge'
        self.tokenizer = BertTokenizer.from_pretrained('../bertlarge')
        self.index = 0   
        self.dev_samples = []
        self.train_samples = []       
        with open('./mrc_data/samplesmrc_dev.json') as f:
            ori_dev_samples = json.loads(f.read())
        with open('./mrc_data/samplesmrc_train.json') as f:
            ori_train_samples = json.loads(f.read())             
        for item in tqdm(ori_dev_samples):
            if item[-2] == 1: 
                self.dev_samples.append(item)               
        for item in tqdm(ori_train_samples):
            if item[-2] == 1:
                self.train_samples.append(item)          
        print(len(self.train_samples))
        self.train_samples = self.train_samples# + neg_train_samples       
        train_samples_len = [len(_[1]) for _ in self.train_samples]
        self.train_samples =(np.array(self.train_samples)[np.argsort(train_samples_len)]).tolist()       
        self.dev_num = len(self.dev_samples)
        self.dev_idxs = list(range(self.dev_num))  
        self.train_num = len(self.train_samples)
        self.train_idxs = list(range(self.train_num)) 
        print(self.dev_num,self.train_num)

        
    def load_dat_new(self, fnm, full=False):
        pairs = []
        qids = []
        spans = []
        labels = []
        f = open(fnm,'r')
        data = json.loads(f.read())
        f.close()
        for item in tqdm(data):       
            question = item[0]
            passage = item[1]
            span = item[2]
            label = item[3]            
            pairs.append((question, passage))                        
            labels.append(label)
            qids.append('neg')
            spans.append(span)           
        print(len(qids),len(pairs),len(labels),sum(labels),fnm)
        return qids, pairs, labels,spans

    def trans_samples(self, qids, pairs, labels=None, spans=None, task_token="[unused3]"):
        max_len = 0
        samples = []
        num = 0   
        for qid, (question, passage), label, span in tqdm(zip(qids, pairs, labels, spans)):            
            question_tok = self.tokenizer.tokenize(question)
            passage_tok = self.tokenizer.tokenize(passage)             
            tokens = ["[CLS]"] + question_tok + ['[SEP]'] + passage_tok
            if label == 1:
                location =[index for index,value in enumerate(tokens) if value == '⇒']
                if len(location) != 2: ipdb.set_trace()
                del(tokens[location[0]])
                del(tokens[location[1]-1])
                span = [location[0],location[1]-2]        
            if len(tokens)>512: num+=1
            if len(tokens) > max_len: max_len = len(tokens)       
            input_type_ids = [0] * (len(question_tok) + 2) + [1] * len(passage_tok)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            samples.append((qid, input_ids, input_type_ids, label, span))         
        print(max_len)
        print(num)
        return samples

    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            smp_idxs = self.train_idxs
            #random.shuffle(smp_idxs)
            samples = self.train_samples
            batch_size = 12
        elif which == 'dev':
            smp_idxs = self.dev_idxs
            samples = self.dev_samples
            batch_size = 8
        elif which == 'test':
            smp_idxs = self.test_idxs
            samples = self.test_samples
            batch_size = 20
        else:
            raise Exception('which should be in [train, dev]!')
        batch_word_idxs, batch_type_idxs, batch_starts, batch_ends, batch_qid = [], [], [], [],[]
        end_idx = smp_idxs[-1]
        batch_labels=[]
        for smp_idx in smp_idxs:            
            smp_infos = samples[smp_idx]
            qid, w_idxs, type_idxs, label,span = smp_infos[:]     
            batch_word_idxs.append(w_idxs)
            batch_type_idxs.append(type_idxs)
            batch_starts.append(span[0])
            batch_ends.append(span[1])
            batch_labels.append(label)
            batch_qid.append(qid)
                        
            if len(batch_word_idxs) == batch_size or smp_idx == end_idx:
                max_len = min(max([len(_) for _ in batch_word_idxs]),512)  
                batch_word_idxs = self.padding_seq(batch_word_idxs, max_len=max_len)
                batch_type_idxs = self.padding_seq(batch_type_idxs, max_len=max_len)
                batch_word_idxs = np.array(batch_word_idxs)
                batch_type_idxs = np.array(batch_type_idxs)
                batch_labels = np.array(batch_labels)
                batch_starts = np.array(batch_starts)
                batch_ends = np.array(batch_ends)
                yield batch_word_idxs, batch_type_idxs, batch_starts, batch_ends, batch_qid,batch_labels
                batch_word_idxs, batch_type_idxs, batch_starts, batch_ends, batch_qid, batch_labels = [], [], [], [], [], []
    
    def padding_seq(self, idxs, max_len=None, pad_unit=0):       
        padded_idxs = []
        for seq in idxs:
            seq = seq[:max_len]
            padding_len = max_len - len(seq)
            for _ in range(padding_len):
                seq.append(pad_unit)
            padded_idxs.append(seq)
        return padded_idxs
    
#===============================================Model=====================================================
class ModelDefine(nn.Module):
    def __init__(self):
        super(ModelDefine, self).__init__()
        pass
        path = '../bertlarge'
        self.bert = BertModel.from_pretrained(path)  # 定义一个模型
        self.sfc = nn.Linear(1024,1)
        self.efc = nn.Linear(1024,1)
        self.drop_out = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, w_idxs1, type_idxs, mask_idxs):        
        embedding_output = self.bert.embeddings(w_idxs1, type_idxs)       
        extended_attention_mask = mask_idxs.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0        
        head_mask = [None] * 24        
        encoded_layers = self.bert.encoder(embedding_output,extended_attention_mask, head_mask)        
        last_layer = encoded_layers[-1]
        start_logits = self.sfc(last_layer)    
        end_logits = self.efc(last_layer)
        return start_logits , end_logits
               
class Model:
    def __init__(self, lr=5e-5, device=None):        
        self.model = ModelDefine() 
        if torch.cuda.is_available():
            self.model.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  
        self.train_loss = AverageMeter()
        self.updates = 0
        opt_layers = list(range(0, 12, 1))
        #self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.dt = DataManager()
        self.flag = 0

    def train(self):
        for i in range(50):        
            print("===" * 10)
            print("epoch%d" % i)
            for batch in tqdm(self.dt.iter_batches(which="train")):
                batch_size = len(batch[0])
                self.model.train()               
                word_idxs, type_idxs, starts, ends = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:-2]]              )
                attention_mask = (word_idxs > 0.5).long().to('cuda')
                start_logits, end_logits = self.model(word_idxs, type_idxs, attention_mask)
                loss = self.loss(start_logits, starts.unsqueeze(-1)) + self.loss(end_logits, ends.unsqueeze(-1))
                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 20)
                self.optimizer.step()
                self.updates += 1
            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset() 
            #with torch.cuda.device('cuda:1'):
            torch.cuda.empty_cache()
            self.validate(epoch=i)
            #self.inference()

               
    def validate(self, which="dev", epoch=-1):
        sigmoid_func = nn.Sigmoid()
        softmax_func = nn.Softmax(dim=-1)
        all_passage_num = 0
        start_correct_num = 0
        end_correct_num = 0
        all_correct_num = 0
        correct_passage_num = 0
        start_correct_num_inla = 0
        end_correct_num_inla = 0
        all_correct_num_inla = 0
        def simple_accuracy(preds1, labels1):
            pe = sum([1.0 if (p1 == p2 and p1 == 1) else 0.0 for p1, p2 in zip(preds1, labels1)]) / sum(preds1)
            re = sum([1.0 if (p1 == p2 and p2 == 1) else 0.0 for p1, p2 in zip(preds1, labels1)]) / sum(labels1)
            correct_num = sum([1.0 if p1 == p2 else 0.0 for p1, p2 in zip(preds1, labels1)])
            f1 = 0 if pe+re == 0 else 2.0*pe*re/(pe+re)
            return f1, correct_num / len(preds1)        

        
        for batch in tqdm(self.dt.iter_batches(which=which)):
            batch_size = len(batch[0])
            self.model.eval()
            word_idxs, type_idxs, starts, ends = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:-2]]
            batch_labels = batch[-1]
            attention_mask = (word_idxs > 0.5).long().to('cuda')
            start_logits, end_logits = self.model(word_idxs, type_idxs, attention_mask)  # 以前这里的输入是有问题的呀。pinyin_flags            
            start_prob = softmax_func(start_logits.squeeze(-1))
            start_pred = np.argmax(start_prob.detach().cpu().numpy(), axis=1).tolist()            
            end_prob = softmax_func(end_logits.squeeze(-1))
            end_pred = np.argmax(end_prob.detach().cpu().numpy(), axis=1).tolist()
            starts = starts.detach().cpu().numpy().tolist()
            ends = ends.detach().cpu().numpy().tolist()
            labels = batch_labels.tolist()
            for sp, ep, s, e, la in zip(start_pred, end_pred, starts, ends, labels):           
                all_passage_num+=1
                if sp==s: start_correct_num += 1
                if ep==e: end_correct_num+=1
                if sp==s and ep==e: all_correct_num+=1
                if la == 1:
                    correct_passage_num +=1
                    if sp==s: start_correct_num_inla += 1
                    if ep==e: end_correct_num_inla+=1
                    if sp==s and ep==e: all_correct_num_inla+=1
                                
        print('passage num ={}'.format(all_passage_num))
        print('positive passage num ={}'.format(correct_passage_num))
        
        print('start correct num = {}'.format(start_correct_num))    
        print('end correct num = {}'.format(end_correct_num))
        print('all correct num = {}'.format(all_correct_num))
        print("start acc={}".format(start_correct_num/all_passage_num))
        print("end acc={}".format(end_correct_num/all_passage_num))
        print("all acc={}".format(all_correct_num/all_passage_num))
        print('inla start correct num = {}'.format(start_correct_num_inla))    
        print('inla end correct num = {}'.format(end_correct_num_inla))
        print('inla all correct num = {}'.format(all_correct_num_inla))
        print("inla start acc={}".format(start_correct_num_inla/correct_passage_num))
        print("inla end acc={}".format(end_correct_num_inla/correct_passage_num))
        print("inla all acc={}".format(all_correct_num_inla /correct_passage_num))        
        #ipdb.set_trace()  
        if which == "dev":
            self.save("mrc_models/mrc_sort_large_check_point_{}.pt".format(epoch), epoch)

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)

    def resume(self, filename):
       # ipdb.set_trace()
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict['network'])
        self.model.to('cuda')
        return self.model


if __name__ == '__main__':
    synonym_model = Model()
    synonym_model.train()

