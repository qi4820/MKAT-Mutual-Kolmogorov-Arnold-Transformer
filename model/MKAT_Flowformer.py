import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,KATEncoder,KATEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, FlowAttention
from layers.Embed import DataEmbedding
import numpy as np

import json
import os

# from kat_rational import KAT_Group
from kat_rational.shared_den_kan import SharedDenKAN


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.label_len = configs.label_len
        self.use_norm = configs.use_norm

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        self.encoder_only = configs.encoder_only

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
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # kan1
        self.embed_kan = SharedDenKAN(configs.d_model)

        # kan2
        self.encoder_kan = SharedDenKAN(configs.d_model)  # --这个是放在Transformer_EncDec.py里面再去做的实例化的


        # Encoder
        self.encoder = KATEncoder(
            [
                KATEncoderLayer(
                    AttentionLayer(
                        FlowAttention(attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    num_groups = configs.num_groups,
                    encoder_kan = self.encoder_kan,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        input_dim = configs.d_model
        self.linear_head = nn.Linear(input_dim, self.pred_len * self.c_out, bias=True)

        if hasattr(self, 'encoder_only') and self.encoder_only:
            configs.distil = False  # 避免序列长度变化
    def only_encoder_forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            means = None
            stdev = None

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if self.use_shared_kan:
            # print("embedding kan")
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D

        enc_out, attns = self.encoder(enc_out, attn_mask=None, global_shared_denominator = self.shared_denominator)

        B = enc_out.shape[0]
        # print("B",B)

        # 5. 取最后一个时间步 (last token)
        last_token = enc_out[:, -1, :]  # [B, d_model]

        # 6. Linear Head: [B, d_model] → [B, pred_len * c_out]
        dec_out = self.linear_head(last_token)

        # 7. Reshape to [B, pred_len, c_out]
        dec_out = dec_out.reshape(B, self.pred_len, self.c_out)

        if self.use_norm:
            stdev_expanded = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1) # [B, pred_len, c_out]
            means_expanded = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * stdev_expanded + means_expanded

        return dec_out  # [B, pred_len, c_out]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm = (x_enc - means) / stdev
        else:
            means = None
            stdev = None
            x_enc_norm = x_enc
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.use_shared_kan:
            # print("embedding encoder")
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D

        enc_out, attns = self.encoder(enc_out, attn_mask=None, global_shared_denominator = self.shared_denominator)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # print("x_enc shape:", x_enc.shape)
        # print("dec_out shape:", dec_out.shape)
        # print("means shape:", means.shape)
        # print("stdev shape:", stdev.shape)
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]
        if self.use_norm:
            stdev_expanded = stdev.repeat(1, self.pred_len, 1)   # [B, 1, C] -> [B, 96, C]
            means_expanded = means.repeat(1, self.pred_len, 1)
            dec_out = dec_out * stdev_expanded + means_expanded

        return dec_out



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.encoder_only:
            # print("encode_only")
            dec_out = self.only_encoder_forecast(x_enc, x_mark_enc)
            # dec_out 已经是 [B, pred_len, c_out]，直接返回！
            return dec_out
        else:
            # print("with decoder!")
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
