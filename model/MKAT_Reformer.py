import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer,KATEncoder,KATEncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import ReformerLayer,FullAttention,AttentionLayer
from layers.Embed import DataEmbedding

import json
import os

# from kat_rational import KAT_Group
from kat_rational.shared_den_kan import SharedDenKAN


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
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
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
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

        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True) # 发现他这个本身就是一个only_encoder的结构，那我们就只做这样的一个就好了，就不给他加decoder，搞不好要出事
            #什么时候我会考虑要加呢？就是在traffic的那个地方出事的时候，但我觉得应该不会出事，这个reformer看起来也像是个重量级的

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            means = None
            stdev = None

        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # 👉 插入点 A：KAN-1（校准原始变量表示）
        if self.use_shared_kan:
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D
            # print("embed kan")

        enc_out, attns = self.encoder(enc_out, attn_mask=None,global_shared_denominator = self.shared_denominator)
        dec_out = self.projection(enc_out)

        # ✅ 关键：先切出预测部分！
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]

        # ✅ 再反归一化（此时 dec_out 时间维 = pred_len）
        if self.use_norm:
            # stdev: [B, 1, c_out], means: [B, 1, c_out]
            # 利用广播机制，无需 repeat！
            dec_out = dec_out * stdev + means  # 自动广播到 [B, pred_len, c_out]

            return dec_out  # [B, L, D]

    def forecast_withDecoder(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            means = None
            stdev = None

        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # 👉 插入点 A：KAN-1（校准原始变量表示）
        if self.use_shared_kan:
            enc_out = self.embed_kan(enc_out, self.shared_denominator)  # ← use shared D

        enc_out, attns = self.encoder(enc_out, attn_mask=None,global_shared_denominator = self.shared_denominator)


        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # print("dec_out after dec embedding",dec_out.shape)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # ✅ 再反归一化（此时 dec_out 时间维 = pred_len）
        if self.use_norm:
            # stdev: [B, 1, c_out], means: [B, 1, c_out]
            # 利用广播机制，无需 repeat！
            dec_out = dec_out * stdev + means  # 自动广播到 [B, pred_len, c_out]

        return dec_out  # [B, L, D]


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.encoder_only:
            # print("only encoder!")
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            # print("现在是encoder-decoder!")
            dec_out = self.forecast_withDecoder(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
