#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:11:56 2019

@author: lin_peihsuan
"""

import torch
import os
import torch.nn as nn
import numpy as np
import time

import CNNmodel
import text2ind

'''先為資料集建立一個字元字典；內含有 "字元 編號 出現次數" 以便作為製造詞embedding的依據'''
word2ind, ind2word = text2ind.get_worddict('/Users/lin_peihsuan/Downloads/22CNN_flex_wordwmbedding/0wordLabel.txt')

'''預測的標籤對應的名稱：0,1'''
label_w2n, label_n2w = text2ind.read_labelFile('/Users/lin_peihsuan/Downloads/22CNN_flex_wordwmbedding/1label.txt')

'''
training data embedding
label embedding(length=5)
0,247,0,0,0,0
1,41,49,0,0,0
0,248,249,0,0,0
1,1,0,0,0,0
0,109,7,0,0,0
'''

'''model parameter'''
textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 1,
    "kernel_size": [1, 2, 3, 4, 5],
    "dropout": 0.2,
}

def get_valData(file):
    datas = open(file, 'r').read().split('\n')
    datas = list(filter(None, datas))
    return datas


def parse_net_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]
    
    return label, score


def main():
    #init net
    print('init net...')
    net = CNNmodel.textCNN(textCNN_param)
    weightFile = 'weight.pkl'
    #weightFile = 'textCNN.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        print('No weight file!')
        exit()
    print(net)
    
    #on GPU
    #net.cuda()
    #net.eval()

    numAll = 0
    numRight = 0
    testData = get_valData('/Users/lin_peihsuan/Downloads/22CNN_flex_wordwmbedding/3test_text_json.txt')
    for data in testData:
        numAll += 1
        data = data.split(',')
        
        label = int(data[0])
        sentence = np.array([int(x) for x in data[1:6]])
        sentence = torch.from_numpy(sentence)
        #on GPU
        #predict = net(sentence.unsqueeze(0).type(torch.LongTensor).cuda()).cpu().detach().numpy()[0]
        predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).cpu().detach().numpy()[0]
        
        label_pre, score = parse_net_result(predict)
        if label_pre == label and score > -100:
            numRight += 1
        #輸出預測準確度五筆為單位輸出
        if numAll % 5 == 0:
            print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))


if __name__ == "__main__":
    main()
