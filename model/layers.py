import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Attention Layer in Transformer (Multi-Head Attention, Feed Forward Network)
# Multi-Head Attention Sublayer
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''
        Arguments:
            * n_head:   number of heads
            * d_model:  hidden dimension of the model
            * d_k:      Query and Key's hidden dimension
            * d_v:      Value's hidden dimension
            * dropout:  dropout rate (only for training)
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # weights (no bias, only weights)
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False) # Query weight [d_model x (H * d_k)]
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False) # Key weight   [d_model x (H * d_k)]
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False) # Value weight [d_model x (H * d_v)]
        self.w_o = nn.Linear(n_head * d_v, d_model, bias=False) # Output projection weight [(H * d_v) x d_model]

        self.dropout = nn.Dropout(dropout)                      # dropout layer (only for the model training step)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)       # layer normalization (layer normalization, used in NLP tasks for better convergence)

    def forward(self, q, k, v, mask=None):
        '''
        Arguments:
            * q: query token embeddings
            * k: key token embeddings
            * v: value token embeddings
            * mask (optional): for masked multihead attention
        '''
        B = q.size(0)           # batch size
        Nq = q.size(1)          # query token length
        N = k.size(1)           # key, value token length
        assert(N == v.size(1))  # assert that key and value have same token length.

        # Generate QKV as well as splitting them into multiple heads.
        Q = self.w_q(q).view(B, Nq, self.n_head, self.d_k)  # B x Nq x H x d_k
        K = self.w_k(k).view(B, N, self.n_head, self.d_k)   # B x N x H x d_k
        V = self.w_v(v).view(B, N, self.n_head, self.d_v)   # B x N x H x d_v

        # Transpose for attention dot product
        # B x H x Nq x d_k (transpose between the head and token length dimension)
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)

        # Multiply Q and KT, then scale. 
        attn = torch.matmul(Q, K.transpose(2,3))/(math.sqrt(self.d_k))

        # Apply the mask before the SoftMax;
        if mask is not None:
            mask = mask.unsqueeze(1)                # unsqueeze mask: [B x Nq x N] -> [B x 1 x Nq x N]
            attn = attn.masked_fill(mask==0, -1e9)  # mask with very very small negative number (-10^9)

        attn = F.softmax(attn, dim=-1)              # row-wise softmax
        attn = self.dropout(attn)                   # attention dropout
        output = torch.matmul(attn, v)              # multiply V to get the output

        # Revert the dimension back to the original for residual connection
        # B x Nq x (H * d_v)
        output = output.transpose(1,2).contiguous().view(B, Nq, -1)
        
        # Output projection
        # B x Nq x d_model
        output = self.w_o(output)

        # Residual Connection
        # B x Nq x d_model
        output += q

        # Layer normalization and return the results.
        # B x Nq x d_model
        return self.layer_norm(output)

## Feed Forward Network Sublayer
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1, activation="ReLU"):
        '''
        Arguments:
            * d_model:  hidden dimension of the model
            * d_ffn:    feed forward network hidden dimension
            * dropout:  dropout rate (only used in training)
            * activation: activation function selection (ReLU by default)
        '''
        super().__init__()
        # Feed Forward Network Weights
        self.w_in = nn.Linear(d_model, d_ffn)   # d_model x d_ffn
        self.w_out = nn.Linear(d_ffn, d_model)  # d_ffn x d_model

        # Layer Normalization Layer
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Activation Layer
        if activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            if activation == "GELU":
                self.activation = nn.GELU()
            else: raise Exception("* ERROR: invalid activation function given.")

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Arguments:
            * x: input (which is the output of the multihead attention layer)
        '''
        # x (from the multihead attention layer): [B x Nq x d_model]
        # out (output): [B x Nq x d_ffn] -> [B x Nq x ffn] -> [B x Nq x d_model]
        # then, add x (input) for residual connection
        out = self.w_out(self.activation(self.w_in)) + x   

        # Layer normalization 
        return self.layer_norm(out)

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, d_k, d_v, dropout=0.1):
        '''
        Arguments:
            * d_model:  hidden dimension of the model
            * d_ffn:    hidden dimension of the feed forward network 
            * n_head:   number of attention heads
            * d_k:      hidden dimension of the query and key
            * d_v:      hidden dimension of the value
            * dropout:  dropout rate (only used in training)
        '''
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn, dropout)

    def forward(self, enc_i, self_attn_mask=None):
        '''
        Arguments
            * enc_i: encoder input
            * self_attn_mask: mask for self attention
        '''
        # Encoder self attention (All Q, K and V are from the same source, enc_i)
        enc_o = self.mha(enc_i, enc_i, enc_i, mask=self_attn_mask)

        # Feed Forward Network
        enc_o = self.ffn(enc_o)
        return enc_o

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, d_k, d_v, dropout=0.1):
        '''
        Arguments:
            * d_model: model hidden dimension
            * d_ffn: feed forward network hidden dimension
            * d_k: key & query hidden dimension
            * d_v: value hiddne dimension
            * dropout: dropout rate
        '''
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.cross_mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ffn, dropout, activation="ReLU")

    def forward(self, dec_i, enc_o, self_attn_mask=None, cross_attn_mask=None):
        '''
        Arguments
            * dec_i: decoder input (autoregressive inputs, which are all the outputs that decoder has generated until now)
            * enc_o: encoder output from the transformer encoder
            * self_attn_mask: attention mask for the decoder's self attention
            * cross_attn_mask: attention mask for the decoder's cross attention (upper triangle is masked out in general)
        '''
        # Decoder self attention (All Q, K and V are from the same source)
        dec_o = self.self_mha(dec_i, dec_i, dec_i, mask=self_attn_mask)

        # Decoder cross attention (Q is from the decoder self attention, while K and V are from the encoder output)
        dec_o = self.cross_mha(dec_o, enc_o, enc_o, mask=cross_attn_mask)

        # Feed Forward Network
        dec_o = self.ffn(dec_o)
        return dec_o


        
