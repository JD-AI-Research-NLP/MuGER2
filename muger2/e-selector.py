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
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import copy
import torch.optim as optim
from train_mrc_large import ModelDefine as MRC_large
from train_mrc_base import ModelDefine as MRC_base


#===============================================Model=====================================================
    
class E_SELECTOR:
    def __init__(self, lr=5e-5, device=None):
        self.flag = 0               
    def selector(self, mode, mrc_mode, retrieval_score_path, prediction_path):
        sigmoid_func = nn.Sigmoid()
        softmax_func = nn.Softmax(-1)
        #=================================================================
        if mrc_mode == 'large':
            mrc_model = MRC_large()
            bert_tokenizer = BertTokenizer.from_pretrained('../bertlarge')
            checkpoint = torch.load("../../trained_models/mrc_large.pt")
        elif  mrc_mode == 'base':
            mrc_model = MRC_base()
            bert_tokenizer = BertTokenizer.from_pretrained('../bertbase')
            checkpoint = torch.load("../../trained_models/mrc_base.pt")
        state_dict = checkpoint['state_dict']
        new_state = {}
        for key in state_dict['network'].keys():
            new_key = key.replace('module.','')
            if new_key != '1':
                new_state[new_key] = state_dict['network'][key]
        mrc_model.load_state_dict(new_state)
        mrc_model.to('cuda')
        mrc_model.eval()      
        #=================================================================
        if mode == 'dev':
            with open('../../processed_data/dev.json') as f:
                all_questions = json.load(f)
            with open('../../processed_data/hop_location_dev.json') as f:
                hop_location = json.load(f)
        elif mode == 'test':
            with open('../../processed_data/hop_location_test.json') as f:
                hop_location = json.load(f)            
        answer4dev = []      
        with open(retrieval_score_path) as f:
            dev_questions = json.loads(f.read())
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
        have_link_q = 0)
        for item in tqdm(all_questions): 
            qid = item['question_id']
            cur_hop_location = hop_location[qid]
            cur_answer={}          
            with open('../../processed_data/WikiTables-WithLinks/request_tok/{}.json'.format(item['table_id'])) as f:
                all_passage = json.load(f) 
            with open('../../processed_data/WikiTables-WithLinks/tables_tok/{}.json'.format(item['table_id'])) as f:
                table = json.load(f)     
            column_len = len(table['data'][0])
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
            anchor_score = np.array(data['anchor_score']).reshape(-1,column_len).tolist()
            answer_score = np.array(data['answer_score']).reshape(-1,column_len).tolist()
            # get global answer score     
            new_answer_score = []
            for i, row_score in enumerate(answer_score):
                cur_row_score = []
                for j, cell_score in enumerate(row_score):
                    new_cell_score = cell_score + data['header_score'][j] + max(anchor_score[i])
                    cur_row_score.append(new_cell_score)
                new_answer_score.append(cur_row_score)      
            new_answer_score = np.array(new_answer_score).reshape(-1).tolist()
            max_answer_score = max(new_answer_score)
            max_answer_score_id = new_answer_score.index(max_answer_score)
            max_answer_context = data['answer_context'][max_answer_score_id]
            max_answer_linktitle = data['answer_linktitle'][max_answer_score_id]
            # get global hop score       
            new_hop_score = []
            for hs, hloc in zip(data['hop_score'], cur_hop_location):
                new_hop_score.append(hs + max(anchor_score[hloc[0]]) + data['header_score'][hloc[1]])        
            max_hop_score = max(new_hop_score)
            max_hop_score_id = new_hop_score.index(max_hop_score)
            max_hop_context = data['hop_context'][max_hop_score_id]
            max_hop_linktitle = data['hop_linktitle'][max_hop_score_id]
            if max_answer_score >= max_hop_score:               
                    ans = max_answer_context       
            else:                    
                    cur_hop_link_c = []
                    cur_hop_link_s = []
                    cur_hop_link_l = []
                    cur_hop_link = max_hop_linktitle
                    for link_s, link_c, link_l in zip(data['link_score'],data['link_context'],data['link_label']):
                        if link_c in cur_hop_link:
                            cur_hop_link_s.append(link_s)
                            cur_hop_link_c.append(link_c)
                            cur_hop_link_l.append(link_l)                         
                    max_cur_hop_link_index = cur_hop_link_s.index(max(cur_hop_link_s)) 
                    max_cur_hop_link_label = cur_hop_link_l[max_cur_hop_link_index]                    
                    max_cur_hop_link_c = cur_hop_link_c[max_cur_hop_link_index]
                    passage_tok = bert_tokenizer.tokenize(all_passage[max_cur_hop_link_c])
                    question_tok = bert_tokenizer.tokenize(item['question'])
                    tokens = ['[CLS]'] + question_tok + ['[SEP]'] + passage_tok
                    input_type_ids = [0] * (len(question_tok) + 2) + [1] * (len(passage_tok))
                    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                    if len(input_ids) > 512: 
                        input_ids = input_ids[0:512]
                        input_type_ids = input_type_ids[0:512]
                    input_ids = torch.from_numpy(np.array(input_ids)).long().to('cuda').unsqueeze(0)
                    input_type_ids = torch.from_numpy(np.array(input_type_ids)).to('cuda').unsqueeze(0)
                    attention_mask = (input_ids > 0.5).long().to('cuda')                                              
                    start_logits, end_logits = mrc_model(input_ids, input_type_ids,attention_mask)                       
                    start_prob = softmax_func(start_logits.squeeze(-1).squeeze(0)).detach().cpu().numpy().tolist()
                    end_prob = softmax_func(end_logits.squeeze(-1).squeeze(0)).detach().cpu().numpy().tolist()
                    span_pos = []
                    span_score = []
                    for i,s in enumerate(start_prob):
                                for j,e in enumerate(end_prob):
                                    if i<=j:
                                        span_score.append(s*e)
                                        span_pos.append([i,j])        
                    span_idx = np.array(span_score).argsort()[::-1]
                    span_pos = np.array(span_pos)[span_idx].tolist()
                    for pos in span_pos:                    
                                sp = pos[0]
                                ep = pos[1]
                                if sp <= ep:                          
                                    answer = tokens[sp:ep+1]
                                    final_answer = []
                                    word_begin = len(final_answer)-1
                                    for i,w in enumerate(answer):
                                        if w[0] != '#':
                                            final_answer.append(w)
                                            word_begin += 1
                                        else:
                                            if word_begin == -1:
                                                final_answer.append(w.replace('##',''))
                                                word_begin += 1
                                            else:
                                                final_answer[word_begin] = final_answer[word_begin] + w.replace('##','')
                                    ans = ' '.join(final_answer)
                                    ans = ans.replace(' . ','.').replace(' , ',',')                                
                                    break
            if ans == ' ':
                ans = max_answer_context
            header_index = data['header_score'].index(max(data['header_score']))
            anchor_index = data['anchor_score'].index(max(data['anchor_score']))
            hop_index = data['hop_score'].index(max(data['hop_score']))
            answer_index = data['answer_score'].index(max(data['answer_score']))
            link_index = data['link_score'].index(max(data['link_score']))
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
            if data['hop_label'][max_hop_score_id] == 1 or data['answer_label'][max_answer_score_id] == 1:
                correct_question_or += 1
            cur_answer['question_id'] = qid
            cur_answer['pred'] = ans          
            answer4dev.append(cur_answer)
        print(len(answer4dev))        
        with open(prediction_path,'w') as f:
            f.write(json.dumps(answer4dev))        
        print('have header q num = {}'.format(have_header_q))
        print('have anchor q num = {}'.format(have_anchor_q))
        print('have hop q num = {}'.format(have_hop_q))
        print('have answer q num = {}'.format(have_answer_q))
        print('have link q num = {}'.format(have_link_q))
        print('============================================')    
        print('correct header num = {}'.format(correct_header))
        print("header acc={}".format(correct_header/have_header_q))
        print('correct anchor num = {}'.format(correct_anchor))
        print("anchor acc={}".format(correct_anchor/(have_anchor_q)))
        print('correct hop num = {}'.format(correct_hop))
        print("hop acc={}".format(correct_hop/have_hop_q))
        print('correct answer num = {}'.format(correct_answer))
        print("answer acc={}".format(correct_answer/have_answer_q))
        print('correct link num = {}'.format(correct_link))
        print("link acc={}".format(correct_link/have_link_q))
        print('correct question num or = {}'.format(correct_question_or)) 
        print("question acc or={}".format(correct_question_or/len(dev_questions)))   
        
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
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        new_state = {}
        for key in state_dict['network'].keys():
            new_key = key.replace('module.','')
            new_state[new_key] = state_dict['network'][key]   
        self.model.load_state_dict(new_state)
        self.model.to('cuda')
        return self.model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='dev')
    parser.add_argument("--mrc_mode", type=str, default="large")
    parser.add_argument("--input_path", type=str, default='./retrieval_score_large.json')
    parser.add_argument("--output_path", type=str, default="./predictions_large.json")    
    args = parser.parse_args()
    synonym_model = E_SELECTOR()
    synonym_model.selector(args.mode, args.mrc_mode, args.input_path, args.output_path)

