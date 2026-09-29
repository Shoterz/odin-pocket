"""Per-request KV cache. Reuses trained weights without modifying the training model."""
import torch
from torch.nn import functional as F


class CachedDecoder:
    def __init__(self,model):
        self.model=model
        self.tokens=[]
        self.cache=[]
        self.device=next(model.parameters()).device

    @staticmethod
    def rotate(attention,x,offset):
        a,b=x[...,::2],x[...,1::2]
        cos=attention.cos[:,:,offset:offset+x.size(2)].to(x.dtype)
        sin=attention.sin[:,:,offset:offset+x.size(2)].to(x.dtype)
        return torch.stack((a*cos-b*sin,a*sin+b*cos),dim=-1).flatten(-2)

    @torch.inference_mode()
    def next_logits(self,ids):
        ids=list(ids[-self.model.config.context:])
        if not ids:
            raise ValueError('Cached decoding needs at least one token')
        incremental=bool(self.cache) and len(ids)==len(self.tokens)+1 and ids[:-1]==self.tokens
        if not incremental:
            self.cache=[]
        offset=len(self.tokens) if incremental else 0
        new_ids=ids[-1:] if incremental else ids
        x=self.model.embedding(torch.tensor([new_ids],device=self.device))
        next_cache=[]
        for i,block in enumerate(self.model.blocks):
            attention=block.attn
            y=block.attn_norm(x)
            q,k,v=attention.qkv(y).view(1,len(new_ids),3,attention.heads,attention.dim).permute(2,0,3,1,4).unbind(0)
            q,k=self.rotate(attention,q,offset),self.rotate(attention,k,offset)
            if incremental:
                past_k,past_v=self.cache[i]
                k,v=torch.cat((past_k,k),dim=2),torch.cat((past_v,v),dim=2)
            next_cache.append((k,v))
            # For one new token all cached keys are past/current; PyTorch's
            # is_causal=True on a rectangular single-query matrix is incorrect.
            y=F.scaled_dot_product_attention(q,k,v,is_causal=not incremental)
            y=y.transpose(1,2).contiguous().view(1,len(new_ids),self.model.config.width)
            x=x+attention.proj(y)
            y=block.ff_norm(x)
            x=x+block.down(F.silu(block.gate(y))*block.up(y))
        self.tokens=ids
        self.cache=next_cache
        return self.model.output(self.model.norm(x[:,-1:]))[0,0]
