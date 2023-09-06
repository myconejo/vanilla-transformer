import math
import torch
import torch.nn as nn
from model.layers import EncoderLayer, DecoderLayer

# Helper Functions
## Gets the pad mask for word sequences
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

## Get the next mask
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# Positional Encoding - encodes positional information to each token
class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        """
        Arguments:
            * d_embedding: embedding dimension
            * dropout: dropout rate
            * max_len: maximum token length
        """
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, 1, d_embedding)
        pe[:, 0, 0::2] = torch.sin(position * div_term) 
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
# Encoder: transformer encoder block
class Encoder(nn.Module):
    '''A transformer encoder with a self-attention and a feed forward network.'''

    def __init__(self, src_vocab_size, d_embedding, n_layers, n_head, d_k, d_v, d_model, d_ffn,
                  pad_index, dropout=0.1, max_len=5000, scale_emb=False):
        super().__init__()
        """
        Arguments:
            * src_vocab_size: source (encoder) vocabulary size
            * d_embedding: embedding hidden dimension
            * n_layers: number of encoder layers
            * n_head: number of heads
            * d_k: Query and Key hidden dimension
            * d_v: Value hidden dimension
            * d_model: model dimension
            * d_ffn: feed forward network hidden dimension
            * pad_index: padding index
            * dropout: dropout rate (default = 0.1)
            * max_len: max token length (default = 5000)
            * scale_emb: if true, then scale the embeddings (default = False)
        """

        # Embedding table
        self.embedding_table = nn.Embedding(src_vocab_size, d_embedding, padding_index=pad_index)

        # Absolute positional encoding (sinusoidal encoding)
        self.pe = PositionalEncoding(d_embedding=d_embedding, dropout=0.1, max_len=max_len)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ffn, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Other parameters
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask):
        """
        Arguments:
            * source_seq: source sequence to be encoded
            * src_mask: mask for the source sequence
        """
        enc_i = self.embedding_table(src_seq)
        if self.scale_emb:
            enc_i *= self.d_model ** 0.5

        enc_i = self.pe(enc_i)
        enc_i = self.dropout(enc_i)
        enc_i = self.layer_norm(enc_i)

        for encoder_layer in self.layers:
            enc_i = encoder_layer(enc_i, self_attn_mask=src_mask)
        
        return enc_i

# Decoder: transformer decoder block 
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_embedding, n_layers, n_head, d_k, d_v, d_model, d_ffn,
                  pad_index, dropout=0.1, max_len=5000, scale_emb=False):
        super().__init__()

        """
        Arguments:
            * trg_vocab_size: target (decoder) vocabulary size
            * d_embedding: embedding hidden dimension
            * n_layers: number of encoder layers
            * n_head: number of heads
            * d_k: Query and Key hidden dimension
            * d_v: Value hidden dimension
            * d_model: model dimension
            * d_ffn: feed forward network hidden dimension
            * pad_index: padding index
            * dropout: dropout rate (default = 0.1)
            * max_len: max token length (default = 5000)
            * scale_emb: if true, then scale the embeddings (default = False)
        """

        # Embedding table
        self.embedding_table = nn.Embedding(trg_vocab_size, d_embedding, padding_index=pad_index)

        # Absolute positional encoding (sinusoidal encoding)
        self.pe = PositionalEncoding(d_embedding=d_embedding, dropout=0.1, max_len=max_len)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ffn, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Layer normalization layer
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Other parameters
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_o, src_mask):
        dec_i = self.embedding_table(trg_seq)
        if self.scale_emb:
            dec_i *= self.d_model ** 0.5

        dec_i = self.pe(dec_i)
        dec_i = self.dropout(dec_i)
        dec_i = self.layer_norm(dec_i)

        for decoder_layer in self.layers:
            dec_i = decoder_layer(dec_i, enc_o, self_attn_mask=trg_mask, cross_attn_mask=src_mask)
        
        return dec_i

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_index, trg_pad_index,
                 d_embedding=512, d_model=512, d_ffn=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, max_len=5000
            ):
        super().__init__()

        """
        Arguments:
            * src_vocab_size: source (encoder) vocabulary size
            * trg_vocab_size: target (decoder) vocabulary size
            * src_pad_index: source sequence padding index
            * trg_pad_index: target sequence padding index
            * d_embedding: embedding table hidden dimension
            * d_model: model hidden dimension
            * d_ffn: feed forward network hidden dimension
            * n_layers: number of transformer encoder/decoder layers
            * n_head: number of heads in multi-head attention
            * d_k, d_v: QKV hidden dimension
            * dropout: dropout rate
            * max_len: maximum length of the token sequence
        """

        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index
        
        self.d_model = d_model

        # Transformer Encoder Block
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            d_embedding=d_embedding,
            n_layers=n_layers,n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_ffn=d_ffn,
            pad_index=src_pad_index, dropout=dropout, 
            max_len=max_len, scale_emb=False
        )

        # Transformer Decoder Block
        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            d_embedding=d_embedding,
            n_layers=n_layers, n_head=n_head, 
            d_k=d_k, d_v=d_v, d_model=d_model, d_ffn=d_ffn,
            pad_index=trg_pad_index, dropout=dropout, 
            max_len=max_len, scale_emb=False
        )

        self.trg_projection = nn.Linear(d_model, trg_vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_embedding

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_o, *_ = self.encoder(src_seq, src_mask)
        dec_o, *_ = self.decoder(trg_seq, trg_mask, enc_o, src_mask)
        
        seq_logit = self.trg_projection(dec_o)
        return seq_logit.view(-1, seq_logit.size(2))