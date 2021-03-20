from functools import cmp_to_key
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        
        # linear weight parameters that split the x in to three parts
        self.tokeys = nn.Linear(k, k * heads, bias=False) # W.T @ x 
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(k*heads, k)

    def forward(self, x, calculation_mode = 'einsum'):

        b, t, k = x.shape
        h = self.heads 

        keys = self.tokeys(x).view(b, t, h, k)       # shape: b, t, h*k --> b, t, h, k
        queries = self.toqueries(x).view(b, t, h, k) # shape: b, t, h*k --> b, t, h, k
        values = self.tovalues(x).view(b, t, h, k)   # shape: b, t, h*k --> b, t, h, k

        if calculation_mode == 'einsum':

            # shape: b t h k
            # shape: b t h k
            # shape: b h t t
            weights = torch.einsum('bthk, bihk -> bhti', [queries, keys]) / torch.sqrt(torch.tensor(k)) # shape: b, h, t, t 
            weights = torch.softmax(weights, dim=-1) # shape: b, h, t, t 

            # shape: b h t t_
            # shape: b t_ h k
            out = torch.einsum('bhte, behk -> bthk', weights, values)# shape: b t h k
            print(out.shape)
            out = out.reshape(b, t, h*k)

        else:

            keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
            queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
            values = values.transpose(1, 2).contiguous().view(b * h, t, k)

            queries = queries / (k ** (1/4))
            keys    = keys / (k ** (1/4))

            # - get dot product of queries and keys, and scale
            dot = torch.bmm(queries, keys.transpose(1, 2))
            # - dot has size (b*h, t, t) containing raw weights
            dot = F.softmax(dot, dim=2) 
            out = torch.bmm(dot, values).view(b, h, t, k)
            out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, calculation_mode='einsum'):
        super().__init__()
        self.calculation_mode = calculation_mode
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        attention = self.attention(x, calculation_mode = self.calculation_mode)
        x = self.norm1( attention + x)
        ff = self.ff(x)
        return self.norm2(ff + x)



class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_len, num_tokens, num_classes, calculation_mode = 'normal'):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_len, k)
        
        self.tbloks = []
        for i in range(depth):
            self.tbloks.append(TransformerBlock(k, heads, calculation_mode='normal'))
        self.tbloks = nn.Sequential(*self.tbloks)

        self.toprobs = nn.Linear(k, num_classes)
    
    def forward(self, x):
        
        embeddings = self.token_emb(x) # shape: b, t, k
        b, t, k = embeddings.shape

        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = embeddings + positions
        x = self.tbloks(x)
        print(x.shape)
        x = self.toprobs(x.mean(dim=1))
        return x 





if __name__ == '__main__':

    atten = SelfAttention(k=256, heads=8)
    transfblock = TransformerBlock(k=256, heads=8, calculation_mode='normal')
    transformer = Transformer( k=256, heads=8, depth=3, seq_len=10, num_tokens=500, num_classes=2)

    inputs = torch.arange(100).reshape(10,10)

    with torch.no_grad():
        # einsum_res = atten(inputs, calculation_mode='einsum')
        # normal_rea = atten(inputs, calculation_mode='normal')
        # output = transformer(inputs)
        output = transformer(inputs)
    

    print(output.shape)

    # print(einsum_res.shape == normal_rea.shape)
    # print((einsum_res == normal_rea).all())
    # print(output.shape)
