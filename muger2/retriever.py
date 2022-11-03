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
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
#===============================================Data=====================================================
class DataManager:
    def __init__(self):         
        self.tokenizer = BertTokenizer.from_pretrained('./bertlarge')
        self.index = 0    
        with open('processed_data/train.json') as f:
            trainq = json.loads(f.read())     
        self.smp4train = {}        
        for item in trainq:
            qid = item['question_id']
            self.smp4train[qid] = {}
            self.smp4train[qid]['true_header'] = []
            self.smp4train[qid]['true_anchor'] = []
            self.smp4train[qid]['true_hop'] = []
            self.smp4train[qid]['true_answer'] = []
            self.smp4train[qid]['true_link'] = []      
            self.smp4train[qid]['false_header'] = []
            self.smp4train[qid]['false_anchor'] = []
            self.smp4train[qid]['false_hop'] = []
            self.smp4train[qid]['false_answer'] = []
            self.smp4train[qid]['false_link'] = []           
        dev_fnm = './processed_data/multitask_dev.json'
        train_fnm = './processed_data/multitask_train.json'     
        dev_qids, dev_pairs, dev_labels, dev_task, dev_context, dev_linktitle = self.load_dat_new(dev_fnm)
        train_qids, train_pairs, train_labels, train_task, train_context, train_linktitle = self.load_dat_new(train_fnm)
        self.dev_samples = self.trans_samples(dev_qids, dev_pairs, dev_labels, dev_task, dev_context, dev_linktitle)
        ori_train_samples = self.trans_samples(train_qids, train_pairs, train_labels, train_task, train_context, train_linktitle)       
        
        for samp in tqdm(ori_train_samples):
            if samp[3] == 1:
                if samp[4] == 0:
                    self.smp4train[samp[0]]['true_header'].append(samp)
                if samp[4] == 1: 
                    self.smp4train[samp[0]]['true_anchor'].append(samp)
                if samp[4] == 2:
                    self.smp4train[samp[0]]['true_hop'].append(samp)
                if samp[4] == 3:
                    self.smp4train[samp[0]]['true_answer'].append(samp)
                if samp[4] == 4:
                    self.smp4train[samp[0]]['true_link'].append(samp)
            else:
                if samp[4] == 0:
                    self.smp4train[samp[0]]['false_header'].append(samp)
                if samp[4] == 1: 
                    self.smp4train[samp[0]]['false_anchor'].append(samp)
                if samp[4] == 2:
                    self.smp4train[samp[0]]['false_hop'].append(samp)
                if samp[4] == 3:
                    self.smp4train[samp[0]]['false_answer'].append(samp)
                if samp[4] == 4:
                    self.smp4train[samp[0]]['false_link'].append(samp)      
        print('all data size:')
        print(len(self.smp4train),len(self.dev_samples))
     
    def resample(self):
        print('sample training data...')
        self.train_samples = {}
        self.train_mask = {}  
        no_answer = 0
        for qid in tqdm(self.smp4train.keys()):           
            self.train_samples[qid] = {}
            self.train_mask[qid] = []
            data = self.smp4train[qid]            
            true_header = len(data['true_header'])
            false_header = len(data['false_header'])
            true_anchor = len(data['true_anchor'])
            false_anchor = len(data['false_anchor'])
            true_hop = len(data['true_hop'])
            false_hop = len(data['false_hop'])
            true_answer = len(data['true_answer'])
            false_answer = len(data['false_answer'])
            true_link = len(data['true_link'])
            false_link = len(data['false_link'])
            #============header===============================
            if true_header >= 1 and false_header >= 5:
                self.train_samples[qid]['header'] = random.sample(data['true_header'], 1) + random.sample(data['false_header'], 5)
                self.train_mask[qid].append(1)
            elif true_header >= 1 and 0 < false_header < 5: 
                self.train_samples[qid]['header'] = random.sample(data['true_header'], 1) + data['false_header']+[data['false_header'][-1]]*(5-false_header)
                self.train_mask[qid].append(1)
            else:
                if true_header + false_header >= 6:
                    self.train_samples[qid]['header'] = random.sample(data['true_header'] + data['false_header'], 6) 
                else:
                    self.train_samples[qid]['header'] = data['true_header'] + data['false_header'] + [(data['true_header'] + data['false_header'])[-1]] * (6 -true_header - false_header)
                self.train_mask[qid].append(0)
            #============anchor===============================
            if true_anchor >= 1 and false_anchor >= 5:
                self.train_samples[qid]['anchor'] = random.sample(data['true_anchor'], 1) + random.sample(data['false_anchor'], 5)
                self.train_mask[qid].append(1)
            elif true_anchor >= 1 and 0 < false_anchor < 5: 
                self.train_samples[qid]['anchor'] = random.sample(data['true_anchor'], 1) + data['false_anchor']+[data['false_anchor'][-1]]*(5-false_anchor)
                self.train_mask[qid].append(1)
            else:
                if true_anchor + false_anchor >= 6:
                    self.train_samples[qid]['anchor'] = random.sample(data['true_anchor'] + data['false_anchor'], 6) 
                else:
                    self.train_samples[qid]['anchor'] = data['true_anchor'] + data['false_anchor'] + [(data['true_anchor'] + data['false_anchor'])[-1]] * (6 -true_anchor - false_anchor)
                self.train_mask[qid].append(0)               
            #============answer===============================
            if true_answer >= 1 and false_answer >= 5:
                self.train_samples[qid]['answer'] = random.sample(data['true_answer'], 1) + random.sample(data['false_answer'], 5)
                self.train_mask[qid].append(1)
            elif true_answer >= 1 and 0 < false_answer < 5: 
                self.train_samples[qid]['answer'] = random.sample(data['true_answer'], 1) + data['false_answer']+[data['false_answer'][-1]]*(5-false_answer)
                self.train_mask[qid].append(1)
            else:
                if true_answer + false_answer >= 6:
                    self.train_samples[qid]['answer'] = random.sample(data['true_answer'] + data['false_answer'], 6) 
                else:
                    self.train_samples[qid]['answer'] = data['true_answer'] + data['false_answer'] + [(data['true_answer'] + data['false_answer'])[-1]] * (6 -true_answer - false_answer)
                self.train_mask[qid].append(0)                
            #============hop===============================
            if true_hop >= 1 and false_hop >= 5:
                self.train_samples[qid]['hop'] = random.sample(data['true_hop'], 1) + random.sample(data['false_hop'], 5)
                self.train_mask[qid].append(1)
            elif true_hop >= 1 and 0 < false_hop < 5: 
                self.train_samples[qid]['hop'] = random.sample(data['true_hop'], 1) + data['false_hop']+[data['false_hop'][-1]]*(5-false_hop)
                self.train_mask[qid].append(1)
            else:
                if true_hop + false_hop >= 6:
                    self.train_samples[qid]['hop'] = random.sample(data['true_hop'] + data['false_hop'], 6) 
                else:
                    self.train_samples[qid]['hop'] = data['true_hop'] + data['false_hop'] + [(data['true_hop'] + data['false_hop'])[-1]] * (6 -true_hop - false_hop)
                self.train_mask[qid].append(0)
            
            #============link===============================
            if true_link >= 1 and false_link >= 5:
                self.train_samples[qid]['link'] = random.sample(data['true_link'], 1) + random.sample(data['false_link'], 5)
                self.train_mask[qid].append(1)
            elif true_link >= 1 and 0 < false_link < 5: 
                self.train_samples[qid]['link'] = random.sample(data['true_link'], 1) + data['false_link']+[data['false_link'][-1]]*(5-false_link)
                self.train_mask[qid].append(1)
            else:
                if true_link + false_link >= 6:
                    self.train_samples[qid]['link'] = random.sample(data['true_link'] + data['false_link'], 6) 
                else:
                    self.train_samples[qid]['link'] = data['true_link'] + data['false_link'] + [(data['true_link'] + data['false_link'])[-1]] * (6 -true_link - false_link)
                self.train_mask[qid].append(0)
        self.dev_num = len(self.dev_samples)
        self.dev_idxs = list(range(self.dev_num))  
        self.train_num = len(self.train_samples)
        self.train_idxs = list(self.train_samples.keys())    
        print(self.dev_num,self.train_num)
        
    def load_dat_new(self, fnm, full=False):
        pairs = []
        labels = []
        qids = []
        task = [] 
        context = [] 
        linktitle = [] 
        f = open(fnm,'r')
        data = json.loads(f.read())
        f.close()
        for qid in tqdm(data.keys()):   
            item = data[qid]
            question = item['question']
            header_data = item['header']
            anchor_data = item['anchor']
            hop_data = item['hop']
            answer_data = item['answer']
            link_data = item['link']         
            for header_item in header_data:
                pairs.append(('header ' + question, header_item[0]))
                labels.append(header_item[-1])
                qids.append(qid)
                task.append(0)
                context.append(header_item[0])
                linktitle.append([])           
            for anchor_item in anchor_data: 
                cell_info = anchor_item[0] + ' [SEP] ' + anchor_item[1] + ' [SEP] ' + anchor_item[2]
                pairs.append(('anchor ' + question, cell_info))                        
                labels.append(anchor_item[-1])
                qids.append(qid)
                task.append(1)
                context.append(anchor_item[3])
                linktitle.append(anchor_item[4])           
            for hop_item in hop_data:
                cell_info = hop_item[0] + ' [SEP] ' + hop_item[1] + ' [SEP] ' + hop_item[2]
                pairs.append(('hop ' + question, cell_info))
                labels.append(hop_item[-1])
                qids.append(qid)
                task.append(2)
                context.append(hop_item[3])
                linktitle.append(hop_item[4])
            for answer_item in answer_data:
                cell_info = answer_item[0] + ' [SEP] ' + answer_item[1] + ' [SEP] ' + answer_item[2]
                pairs.append(('answer ' + question, cell_info))
                qids.append(qid)
                task.append(3)
                labels.append(answer_item[-1])
                context.append(answer_item[3])
                linktitle.append(answer_item[4])
            for link_item in link_data:
                pairs.append(('link ' + question, link_item[0]))
                labels.append(link_item[-1])
                qids.append(qid)
                task.append(4)
                context.append(link_item[1])
                linktitle.append([])         
        print(len(qids),len(pairs),len(labels),len(task),len(context),len(linktitle),sum(labels),fnm)
        return qids, pairs, labels, task, context, linktitle

    def trans_samples(self, qids, pairs, labels, task, context, linktitle, task_token="[unused3]"):
        samples = []      
        for qid, (question, info), label, t , con, lin in tqdm(zip(qids, pairs, labels, task, context, linktitle)):          
            question_tok = self.tokenizer.tokenize(question)
            info_tok = self.tokenizer.tokenize(info)  
            tokens = ["[CLS]"] + question_tok + ['[SEP]'] + info_tok + ['[SEP]']            
            input_type_ids = [0] * (len(question_tok) + 2) + [1] * (len(info_tok)+1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            samples.append((qid, input_ids, input_type_ids, label, t, con, lin))            
        return samples

    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            smp_idxs = self.train_idxs
            random.shuffle(smp_idxs)
            samples = self.train_samples
            masks = self.train_mask
            batch_size = 1
        elif which == 'dev':
            smp_idxs = self.dev_idxs
            samples = self.dev_samples
            batch_size = 16
        elif which == 'test':
            smp_idxs = self.test_idxs
            samples = self.test_samples
            batch_size = 1
        else:
            raise Exception('which should be in [train, dev]!')
        batch_word_idxs, batch_type_idxs, batch_labels, batch_qid, batch_task, batch_context, batch_linktitle, batch_mask = [], [], [], [], [],[],[],[]
        end_idx = smp_idxs[-1]        
        if which == 'train':
          for smp_idx in smp_idxs:   
            smp_infos = samples[smp_idx]
            smp_mask = masks[smp_idx] 
            smp_infos_all = smp_infos['header'] + smp_infos['anchor'] + smp_infos['answer']  + smp_infos['hop'] + smp_infos['link']
            qid = [_[0] for _ in smp_infos_all]
            w_idxs = [_[1] for _ in smp_infos_all]
            type_idxs = [_[2] for _ in smp_infos_all]
            label = [_[3] for _ in smp_infos_all]
            task = [_[4] for _ in smp_infos_all]
            context = [_[5] for _ in smp_infos_all]
            linktitle = [_[6] for _ in smp_infos_all]                
            batch_word_idxs.append(w_idxs)
            batch_type_idxs.append(type_idxs)
            batch_labels.append(label)
            batch_qid.append(qid)
            batch_task.append(task)
            batch_context.append(context)
            batch_linktitle.append(linktitle)
            batch_mask.append(smp_mask)                       
            if len(batch_word_idxs) == batch_size or smp_idx == end_idx:
                lens = []
                for x in batch_word_idxs:
                    for y in x:
                        lens.append(len(y))
                max_len = min(max(lens),512)  
                batch_word_idxs = self.padding_seq4train(batch_word_idxs, max_len=max_len)
                batch_type_idxs = self.padding_seq4train(batch_type_idxs, max_len=max_len)
                batch_labels = np.array(batch_labels)
                batch_word_idxs = np.array(batch_word_idxs)
                batch_type_idxs = np.array(batch_type_idxs)
                batch_task = np.array(batch_task)
                batch_mask = np.array(batch_mask)
                yield batch_word_idxs, batch_type_idxs, batch_labels, batch_task, batch_qid, batch_context, batch_linktitle, batch_mask
                batch_word_idxs, batch_type_idxs, batch_labels, batch_qid, batch_task, batch_context, batch_linktitle, batch_mask = [], [], [], [], [],[],[],[]
        else:
          for smp_idx in smp_idxs:   
            smp_infos = samples[smp_idx]
            qid, w_idxs, type_idxs, label,task,context,linktitle = smp_infos[:]      
            batch_word_idxs.append(w_idxs)
            batch_type_idxs.append(type_idxs)
            batch_labels.append(label)
            batch_qid.append(qid)
            batch_task.append(task)
            batch_context.append(context)
            batch_linktitle.append(linktitle)                        
            if len(batch_word_idxs) == batch_size or smp_idx == end_idx:
                max_len = min(max([len(_) for _ in batch_word_idxs]),512)  
                batch_word_idxs = self.padding_seq4dev(batch_word_idxs, max_len=max_len)
                batch_type_idxs = self.padding_seq4dev(batch_type_idxs, max_len=max_len)
                batch_labels = np.array(batch_labels)
                batch_word_idxs = np.array(batch_word_idxs)
                batch_type_idxs = np.array(batch_type_idxs)
                batch_task = np.array(batch_task)
                yield batch_word_idxs, batch_type_idxs, batch_labels, batch_task, batch_qid, batch_context, batch_linktitle
                batch_word_idxs, batch_type_idxs, batch_labels, batch_qid, batch_task, batch_context, batch_linktitle = [], [], [], [], [],[],[]

    
    def padding_seq4dev(self, idxs, max_len=None, pad_unit=0):       
        padded_idxs = []
        for seq in idxs:
            seq = seq[:max_len]
            padding_len = max_len - len(seq)
            for _ in range(padding_len):
                seq.append(pad_unit)
            padded_idxs.append(seq)
        return padded_idxs
                
    def padding_seq4train(self, idxs, max_len=None, pad_unit=0):      
        padded_idxs = []
        for group in idxs:
            for seq in group:
                seq = seq[:max_len]
                padding_len = max_len - len(seq)
                for _ in range(padding_len):
                    seq.append(pad_unit)
                padded_idxs.append(seq)
        return padded_idxs
    
#===============================================Model=====================================================
class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        pass        
        self.model = ModelDefine()
        self.consfc = nn.Linear(1024*2,1)
          
    def forward(self, w_idxs1, type_idxs, mask_idxs):
        hiddens = hiddens.view(5,-1,1024)
        positive = hiddens[:,0,:].unsqueeze(1).repeat(1,6,1)
        hidden2contrast = torch.cat([hiddens,positive],dim=-1)
        hidden2contrast = self.consfc(hidden2contrast)
        return logits, hidden2contrast
                
class ModelDefine(nn.Module):
    def __init__(self):
        super(ModelDefine, self).__init__()
        pass
        self.bert = BertModel.from_pretrained('./bertlarge')  
        self.fc = nn.Linear(1024,1)
        self.drop_out = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
                
    def forward(self, w_idxs1, type_idxs, mask_idxs):        
        embedding_output = self.bert.embeddings(w_idxs1, type_idxs)
        extended_attention_mask = None        
        head_mask = [None] * 24        
        encoded_layers = self.bert.encoder(embedding_output,extended_attention_mask, head_mask)        
        last_layer = encoded_layers[-1]
        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2).contiguous(), kernel_size=last_layer.size(1)).squeeze(-1)        
        logits = self.fc(max_pooling_fts)             
        return logits, max_pooling_fts
                  
class Model:
    def __init__(self, lr=5e-5, device=None):
        self.submodel = SubModel() 
        if torch.cuda.is_available():
            self.submodel.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.submodel.model = nn.DataParallel(self.submodel.model)  
        self.train_loss = AverageMeter()
        self.updates = 0
        opt_layers = list(range(0, 12, 1))
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.CrossEntropyLoss(reduction = 'none')
        self.optimizer = optim.Adamax([p for p in self.submodel.parameters() if p.requires_grad], lr=lr)
        self.dt = DataManager()
        self.dt.resample()
        self.flag = 0        

    def train(self):
        for i in range(50):
            if i>0:
                self.dt.resample()
            print("===" * 10)
            print("epoch%d" % i)
            for batch in tqdm(self.dt.iter_batches(which="train")):
                batch_size = len(batch[0])
                self.submodel.train()                               
                word_idxs, type_idxs, labels = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:3]]  
                task_mask = Variable(torch.from_numpy(batch[-1])).float().cuda() 
                attention_mask = (word_idxs > 0.5).long().to('cuda')
                labels = labels.transpose(0,1).float()
                logits, hiddens = self.submodel(word_idxs, type_idxs, attention_mask)
                loss_bce = self.loss1(logits, labels)            
                labels = labels.view(5,-1)
                _,cos_labels = torch.max(labels,dim=-1)                         
                if task_mask.sum(-1) != 0:
                    loss_cl = (self.loss2(hiddens.squeeze(-1), cos_labels) * task_mask).sum(-1) / task_mask.sum(-1)
                    loss = loss_bce + loss_cl
                else:
                    loss = loss_bce
                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 20)
                self.optimizer.step()
                self.updates += 1
            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset() 
            #torch.cuda.empty_cache()
            self.validate(epoch=i)
                      
    def validate(self, which="dev", epoch=-1):
        sigmoid_func = nn.Sigmoid()
        dev_questions = {}         
        all_header_num =0
        all_anchor_num = 0
        all_hop_num = 0
        all_answer_num = 0
        all_link_num = 0
        answer4dev = []                      
        for batch in tqdm(self.dt.iter_batches(which=which)):
            batch_size = len(batch[0])
            self.submodel.eval()
            word_idxs, type_idxs, labels = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:3]]
            batch_task = batch[3]
            batch_qid = batch[4]
            batch_context = batch[5]
            batch_linktitle = batch[6]
            attention_mask = (word_idxs > 0.5).long().to('cuda')
            logits,_ = self.submodel.model(word_idxs, type_idxs,attention_mask)  
            score = sigmoid_func(logits.squeeze(-1)).detach().cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()
            for qid, ps, la, t, con, lin in zip(batch_qid, score, labels, batch_task, batch_context, batch_linktitle): 
                if qid not in dev_questions.keys():
                    dev_questions[qid] = {}
                    dev_questions[qid]['header_score'] = []
                    dev_questions[qid]['anchor_score'] = []
                    dev_questions[qid]['hop_score'] = []
                    dev_questions[qid]['answer_score'] = []
                    dev_questions[qid]['link_score'] = []                    
                    dev_questions[qid]['header_label'] = []
                    dev_questions[qid]['anchor_label'] = []
                    dev_questions[qid]['hop_label'] = []
                    dev_questions[qid]['answer_label'] = []
                    dev_questions[qid]['link_label'] = []                    
                    dev_questions[qid]['header_context'] = []
                    dev_questions[qid]['anchor_context'] = []
                    dev_questions[qid]['hop_context'] = []
                    dev_questions[qid]['answer_context'] = []
                    dev_questions[qid]['link_context'] = []                                       
                    dev_questions[qid]['anchor_linktitle'] = []
                    dev_questions[qid]['hop_linktitle'] = []
                    dev_questions[qid]['answer_linktitle'] = []                                       
                if t == 0:
                    all_header_num += 1
                    dev_questions[qid]['header_score'].append(ps)
                    dev_questions[qid]['header_label'].append(la)
                    dev_questions[qid]['header_context'].append(con)                    
                if t == 1:
                    all_anchor_num += 1
                    dev_questions[qid]['anchor_score'].append(ps)
                    dev_questions[qid]['anchor_label'].append(la)   
                    dev_questions[qid]['anchor_context'].append(con)
                    dev_questions[qid]['anchor_linktitle'].append(lin)                    
                if t == 2:
                    all_hop_num += 1                   
                    dev_questions[qid]['hop_score'].append(ps)
                    dev_questions[qid]['hop_label'].append(la) 
                    dev_questions[qid]['hop_context'].append(con)
                    dev_questions[qid]['hop_linktitle'].append(lin)                   
                if t == 3:
                    all_answer_num += 1                   
                    dev_questions[qid]['answer_score'].append(ps)
                    dev_questions[qid]['answer_label'].append(la) 
                    dev_questions[qid]['answer_context'].append(con)
                    dev_questions[qid]['answer_linktitle'].append(lin)                   
                if t == 4:
                    all_link_num += 1
                    dev_questions[qid]['link_score'].append(ps)
                    dev_questions[qid]['link_label'].append(la) 
                    dev_questions[qid]['link_context'].append(con)       
        print('all header num ={}'.format(all_header_num))
        print('all anchor num ={}'.format(all_anchor_num))
        print('all hop num ={}'.format(all_hop_num))
        print('all answer num ={}'.format(all_answer_num))
        print('all link num ={}'.format(all_link_num))        
        print('dev_questions num = {}'.format(len(dev_questions)))   
        with open('retrieval_scores_large_{}.json'.format(epoch),'w') as f:
            f.write(json.dumps(dev_questions))
        correct_question_or = 0
        correct_header = 0
        correct_anchor = 0
        correct_hop = 0
        correct_answer = 0
        correct_link = 0               
        have_header_q = 0
        have_anchor_q = 0
        have_hop_q = 0
        have_answer_q = 0
        have_link_q = 0        
        for qid in tqdm(dev_questions.keys()):  
            data = dev_questions[qid]
            if sum(data['header_label']) != 0:
                have_header_q+=1
            if sum(data['anchor_label']) != 0:
                have_anchor_q+=1
            if sum(data['hop_label']) != 0:
                have_hop_q+=1
            if sum(data['answer_label']) != 0:
                have_answer_q+=1
            if sum(data['link_label']) != 0:
                have_link_q+=1           
            header_index = data['header_score'].index(max(data['header_score']))
            anchor_index = data['anchor_score'].index(max(data['anchor_score']))
            hop_index = data['hop_score'].index(max(data['hop_score']))
            link_index = data['link_score'].index(max(data['link_score']))
            answer_index = data['answer_score'].index(max(data['answer_score']))
            if data['header_label'][header_index]==1:                
                correct_header += 1
            if data['anchor_label'][anchor_index] == 1:
                correct_anchor += 1
            if data['hop_label'][hop_index] == 1:
                correct_hop += 1
            if data['answer_label'][answer_index] == 1:
                correct_answer += 1
            if data['link_label'][link_index] == 1:
                correct_link += 1
            if data['hop_label'][hop_index] == 1 or data['answer_label'][answer_index] == 1:
                correct_question_or += 1
        
        print('have header q num = {}'.format(have_header_q))
        print('have anchor q num = {}'.format(have_anchor_q))
        print('have hop q num = {}'.format(have_hop_q))
        print('have answer q num = {}'.format(have_answer_q))
        print('have link q num = {}'.format(have_link_q))
        print('============================================')    
        print('correct header num = {}'.format(correct_header))
        print("header acc={}".format(correct_header/have_header_q))
        print('correct anchor num = {}'.format(correct_anchor))
        print("anchor acc={}".format(correct_anchor/have_anchor_q))
        print('correct hop num = {}'.format(correct_hop))
        print("hop acc={}".format(correct_hop/have_hop_q))
        print('correct answer num = {}'.format(correct_answer))
        print("answer acc={}".format(correct_answer/have_answer_q))
        print('correct link num = {}'.format(correct_link))
        print("link acc={}".format(correct_link/have_link_q))
        print('correct question num or = {}'.format(correct_question_or)) 
        print("question acc or={}".format(correct_question_or/len(dev_questions)))
        if which == "dev":
            self.save("mrc_models/multitask_finalcontrast_large_check_point_{}.pt".format(epoch), epoch)
    
    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.submodel.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)

    def resume(self, filename):        
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        self.submodel.load_state_dict(state_dict['network'])
        self.submodel.to('cuda')
        return self.submodel

if __name__ == '__main__':
    synonym_model = Model()
    synonym_model.train()
#    synonym_model.validate()

