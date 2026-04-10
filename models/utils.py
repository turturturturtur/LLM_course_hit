import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


# class ScaledDotProductAttention(nn.Module):
#     '''
#     Args:
#         We believe that d_model = d_k = d_v, so we can write as follows:
#     '''
#     def __init__(self,d_model):
#         super().__init__()
#         self.d_k = d_model
#         self.w_q = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
#         self.w_k = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
#         self.w_v = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
#     def forward(self,x):
#         q = self.w_q(x)
#         k = self.w_k(x)
#         v = self.w_v(x)
#         score = q @ torch.transpose(k,-1,-2) / math.sqrt(self.d_k)
#         attention = F.softmax(score,-1) @ v
#         return attention

# we save q,k,v in multiHeadAttention to save GPU memories.
class ScaledDotProductAttention(nn.Module):
    '''
    Args:
        We believe that d_model = d_k = d_v, so we can write as follows:
    '''
    def __init__(self):
        super().__init__()
        # self.d_k = d_model
        # self.w_q = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
        # self.w_k = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
        # self.w_v = nn.Linear(in_features=self.d_k, out_features=self.d_k, bias=False)
    def forward(self,q,k,v,mask=None):
        # q = self.w_q(x)
        # k = self.w_k(x)
        # v = self.w_v(x)
        d_k = q.shape[-1]
        score = q @ torch.transpose(k,-1,-2) / math.sqrt(d_k)
        
        # in this way, we use mask to mask the meanless word like [cls]
        if mask is not None: 
            score = score.masked_fill(mask==0,1e-9) # give a small value to prevent softmax divided by zero(if all place in mask=0)
        
        attn_weights = F.softmax(score,-1)
        attention = attn_weights @ v
        return attention, attn_weights


class MultiHeadAttention(nn.Module):
    """
    The only different between singleHead and multiHead is to cut the d_model dim.
    The shape of the input x is: [batch_size, seq_len, token_dim(d_model)]
    The flow of tensor can be written as:
    [batch_size, seq_len, token_dim(d_model)] -> [batch_size, seq_len, num_head, token_dim(d_k)] -> [batch_size, seq_len, token_dim(d_model)]
    Args:
        We assume that d_model = num_head * d_k, so we can write as follows:
    """
    def __init__(self,d_model,num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = self.d_model // self.num_head
        assert self.d_model == self.d_k * self.num_head

        # When we want to calculate attention, we actually need to calculate seq_len and d_k
        self.w_q = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.w_k = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.w_v = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False)
        self.w_o = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=False) # We use this one to combine each head

        # single head attention
        self.sdpa = ScaledDotProductAttention()
        
    def forward(self,x,mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size,seq_len,self.num_head,self.d_k).transpose(1,2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1,2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1,2)
        
        if mask is not None:
            # reshape [batch, seq_len] -> [batch, 1, 1, seq_len] to broadcast over heads and query positions
            mask = mask.unsqueeze(1).unsqueeze(2)

        attention, attn_weights = self.sdpa(q,k,v,mask)

        # transpose num_head and seq_len
        context = attention.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        output = self.w_o(context)

        return output, attn_weights
