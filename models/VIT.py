import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import copy
import argparse
import sys
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


class VIT(nn.Module):

    def __init__(self, img_size=(224, 224), inp_ch = 3, patch_size=(16, 16), emb_dim=128,
                 mlp_dim=128, num_heads=16, num_layers=10, n_classes=2,
                 dropout_rate=0., at_d_r=0.0):
        super(VIT, self).__init__()

        self.nl = num_layers
        ih, iw = img_size
        ph, pw = patch_size
        num_patches = int((ih * iw) / (ph * pw))
        self.cls_tokens = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.patch_embed = Conv2d(in_channels=inp_ch,
                                  out_channels=emb_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.tel = nn.ModuleList()
        for i in range(num_layers):
            layer = transencoder(emb_dim, mlp_dim, num_heads, at_d_r)
            self.tel.append(layer)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        cls_token = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        embeddings = x + self.pos_embed
        embeddings = self.dropout(embeddings)
        for layer in self.tel:
            enc = layer(embeddings)
        mlp_head = self.mlp_head(enc[:, 0])
        return mlp_head


class transencoder(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads, at_d_r):
        super(transencoder, self).__init__()

        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.mha = mha(emb_dim, num_heads, at_d_r)
        self.mlp = Mlp(emb_dim, mlp_dim)

    def forward(self, x):
        n = self.norm(x)
        attn = self.mha(n, n, n)
        output = attn + x
        n2 = self.norm(output)
        ff = self.mlp(n2)
        out = ff + output
        return out


class mha(nn.Module):
    def __init__(self, h_dim, n_heads, at_d_r):
        super().__init__()
        self.h_dim = h_dim
        self.linear = nn.Linear(h_dim, h_dim, bias=False)
        self.num_heads = n_heads
        self.norm = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(at_d_r)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        rs = q.size()[0]
        batches, sequence_length, embeddings_dim = q.size()
        q1 = nn.ReLU()(self.linear(q))
        k1 = nn.ReLU()(self.linear(k))
        v1 = nn.ReLU()(self.linear(v))

        q2 = torch.cat(torch.chunk(q1, self.num_heads, dim=2), dim=0)
        k2 = torch.cat(torch.chunk(k1, self.num_heads, dim=2), dim=0)
        v2 = torch.cat(torch.chunk(v1, self.num_heads, dim=2), dim=0)

        outputs = torch.bmm(q2, k2.transpose(2, 1))
        outputs = outputs / (k2.size()[-1] ** 0.5)
        outputs = F.softmax(outputs, dim=-1)
        outputs = self.dropout(outputs)
        outputs = torch.bmm(outputs, v2)
        outputs = outputs.split(rs, dim=0)
        outputs = torch.cat(outputs, dim=2)
        outputs += outputs + q
        outputs = self.norm(outputs)
        return outputs


class Mlp(nn.Module):
    def __init__(self, emb_dim, mlp_dim, dropout_rate=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(emb_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

# model = VIT()
