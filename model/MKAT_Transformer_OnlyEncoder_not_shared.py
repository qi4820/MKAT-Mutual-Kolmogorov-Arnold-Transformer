import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, KATEncoder, KATEncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
import os
import json

# KAT
from kat_rational.shared_den_kan import SharedDenKAN

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # === 加载 KAN 初始化参数 ===
        cfd = os.path.dirname(os.path.realpath(__file__))
        print(cfd)
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

        self.shared_denominator_encoder = nn.Parameter(
            w_den.repeat(configs.d_model, 1).float()  # [d_model, Q]
        )

        # use_kan
        self.use_shared_kan = configs.use_shared_kan

        # kan1
        self.embed_kan = SharedDenKAN(configs.d_model)

        # kan2
        self.encoder_kan = SharedDenKAN(configs.d_model)  # --这个是放在Transformer_EncDec.py里面再去做的实例化的

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
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
        # Decoder
        # self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # )
        self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)
        self.sequence_projection = nn.Linear(self.seq_len, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化（使用全局统计量）
        # === 归一化（iTransformer style）===
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            means = None
            stdev = None

        # After embedding
        # 👉 插入点 A：KAN-1（校准原始变量表示）
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if self.use_shared_kan:
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D
            # print("已经使用了embed_kan")

        enc_out, attns = self.encoder(enc_out, attn_mask=None, global_shared_denominator = self.shared_denominator_encoder)

        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # print("enc_out",enc_out.shape)
        # dec_out = self.projector(enc_out)
        # print("dec_out",dec_out.shape)
        # print("stdev",stdev.shape)
        # print("meas",means.shape)
        # Only-Encoder Projection
        enc_out = self.projection(enc_out)          # [B, 96, 512] -> [B, 96, 21]
        enc_out = enc_out.permute(0, 2, 1)          # [B, 21, 96]
        dec_out = self.sequence_projection(enc_out) # [B, c_out, L] -> [B, c_out, pred_len]
        dec_out = dec_out.permute(0, 2, 1)          # [B, pred_len, c_out]
        # Per-variable prediction
         # === 反归一化 ===
        if self.use_norm:
            stdev_expanded = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            means_expanded = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * stdev_expanded + means_expanded



        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
