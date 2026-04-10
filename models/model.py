from models.utils import MultiHeadAttention
import torch
import torch.nn as nn
import os
import numpy as np
import math
from .utils import MultiHeadAttention


class MedicalModel(nn.Module):
    def __init__(
        self, vocab_size, d_model, d_hidden, max_len, num_classes, num_head, num_layer, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.num_head = num_head
        self.d_hidden = d_hidden
        self.num_layer = num_layer
        self.dropout = dropout
        self.num_classes = num_classes

        self.word_ebd = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.d_model)
        self.position_ebd = PostionalEncode(max_len=self.max_len,d_model=self.d_model)
        self.encoders = nn.ModuleList([
                    EncoderLayer(d_model=self.d_model, d_hidden=self.d_hidden, num_head=self.num_head, dropout=self.dropout) 
                    for _ in range(self.num_layer)
                ])

        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self,x,mask=None):
        x = self.word_ebd(x) # [batch, seq_len, d_model]
        x = self.position_ebd(x)  # [batch, seq_len, d_model]
        all_attn_weights = []
        for layer in self.encoders:
            x, attn_weights = layer(x, mask)  # [batch, seq_len, d_model]
            all_attn_weights.append(attn_weights)

        cls_output = x.mean(1) # [batch, d_model]
        logits = self.classifier(cls_output) # [batch,num_classes]
        return logits, all_attn_weights  # [batch,num_classes]


class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_hidden,num_head,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.multihead_attn = MultiHeadAttention(d_model=self.d_model,num_head=self.num_head)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model,self.d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_hidden,self.d_model))
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self,x,mask=None):
        attn_out, attn_weights = self.multihead_attn(self.layernorm1(x), mask=mask)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.layernorm2(x)))
        return x, attn_weights


class PostionalEncode(nn.Module):
    '''
    Actually, we just translate the math formulation to matrix.
    '''
    def __init__(self,max_len,d_model):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(self.max_len,self.d_model) # [max_len, d_model]
        pos = torch.arange(0,self.max_len,1).unsqueeze(-1)
        _2i = torch.arange(0,self.d_model,2)
        inv_freq = 1 / (1e4 ** (_2i/self.d_model)).unsqueeze(0)
        pe[:, 0::2] = torch.sin(pos * inv_freq)
        pe[:, 1::2] = torch.cos(pos * inv_freq)
    
        pe = pe.unsqueeze(0) # [1, max_len,d_model]
        self.register_buffer("pe",pe)
        
        
        

    def forward(self,x):
        '''
        x is [batch, seq_len, d_model]
        
        We add x in it is just transform it into a layer whose input is x and add it with pe automatically
        
        And we have observed that if we put pe, pos, _2i matrix in forward(), torch will create it when we call this class,
        which will cause great waste of memories
        '''
        # pe = torch.zeros(self.max_len,self.d_model) # [batch, seq_len, d_model]
        # pos = torch.arange(0,self.max_len,1).unsqueeze(-1)
        # _2i = torch.arange(0,self.d_model,2)
        # inv_freq = 1 / (1e4 ** (_2i/self.d_model)).unsqueeze(0)
        # pe[:, 0::2] = torch.sin(pos * inv_freq)
        # pe[:, 1::2] = torch.cos(pos * inv_freq)
        input_len = x.shape[1]
        x = x + self.pe[:,:input_len,:]
        return x
