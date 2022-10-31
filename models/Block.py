import mindspore.nn as nn
from mindspore import ops

from models.drop_path import DropPath
from models.mlp import Mlp


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0-attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0-proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        batchmatual = ops.BatchMatMul(transpose_b=True)
        attn = batchmatual(q, k) * self.scale
        softmax = nn.Softmax()
        attn = softmax(attn)
        attn = self.attn_drop(attn)

        batchmatual2=ops.BatchMatMul()
        x=batchmatual2(attn,v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #print(222222)
        #print(dim)
        self.norm1 = norm_layer((dim,), epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #print("drop_path",drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()
        #print("self.drop_path",self.drop_path)
        self.norm2 = norm_layer((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #print(x)
        return x


