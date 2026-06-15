import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder,KATEncoder, EncoderLayer, KATEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import os
import json

# from kat_rational import KAT_Group
from kat_rational.shared_den_kan import SharedDenKAN


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # === 加载 KAN 初始化参数 ===
        cfd = os.path.dirname(os.path.realpath(__file__))
        # print(cfd)
        try:
            with open(f'{cfd}/init.json') as f:
                init_data = json.load(f)
            w_num = torch.tensor(init_data["gelu"]["init_w_numerator"])  # shape: [K]
            w_den = torch.tensor(init_data["gelu"]["init_w_denominator"])  # shape: [Q]
            # 在 __init__ 方法中，加载 init_data 后添加：
            print("Successfully loaded init.json!")
            print(f"GELU numerator: {w_num.tolist()}")
            print(f"GELU denominator: {w_den.tolist()}")
        except Exception as e:
            print(f"Warning: Failed to load init.json, using random init. Error: {e}")
            w_num = torch.randn(6)
            w_den = torch.randn(4)

        # Define ONE shared denominator for the entire d_model space
        self.shared_denominator = nn.Parameter(
            w_den.repeat(configs.d_model, 1).float()  # [d_model, Q]
        )

        # use_kan
        self.use_shared_kan = configs.use_shared_kan

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # kan1
        self.embed_kan = SharedDenKAN(configs.d_model)

        # kan2
        self.encoder_kan = SharedDenKAN(configs.d_model)  # --这个是放在Transformer_EncDec.py里面再去做的实例化的

        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = KATEncoder(
            [
                KATEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    num_groups = configs.num_groups,
                    encoder_kan = self.encoder_kan,
                    only_modular_kan = configs.only_modular_kan,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # After embedding
        # 👉 插入点 A：KAN-1（校准原始变量表示）
        if self.use_shared_kan:
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D
            # print("有使用embed_kan")

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None, global_shared_denominator = self.shared_denominator)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates


        # print("dec_out",dec_out.shape)
        # print("stdev",stdev.shape)
        # print("meas",means.shape)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]