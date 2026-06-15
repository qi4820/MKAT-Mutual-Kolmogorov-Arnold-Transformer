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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化 x_enc（在时间维度上，每个样本、每个通道独立归一化）
        # x_enc: [B, L_in, D]
        means = x_enc.mean(1, keepdim=True)        # [B, 1, D]
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, D]

        x_enc_norm = (x_enc - means) / stdev

        # Step 2: 用相同的统计量归一化 x_dec（避免未来信息泄露）
        # x_dec: [B, L_out_total, D]，通常前 label_len 是真实值，后 pred_len 是占位符（如 zeros）
        x_dec_norm = (x_dec - means) / stdev
        # After embedding
        # 👉 插入点 A：KAN-1（校准原始变量表示）
        # Embedding
        #print("x_enc",x_enc.shape)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print("enc_out after embedding",enc_out.shape)

        if self.use_shared_kan:
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D
            # print("已经使用了embed_kan")

        # print("enc_out before encoder",enc_out.shape)

        enc_out, attns = self.encoder(enc_out, attn_mask=None, global_shared_denominator = self.shared_denominator)

        # print("enc_out after encoder",enc_out.shape)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # print("dec_out after dec embedding",dec_out.shape)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # print("dec_out after decoder",dec_out.shape)

        # 反归一化（恢复原始尺度）
        dec_out = dec_out * stdev + means

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
