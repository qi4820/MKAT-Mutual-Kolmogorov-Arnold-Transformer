import torch.nn as nn
import torch.nn.functional as F
import torch

from kat_rational import KAT_Group
from kat_rational.shared_den_kan import SharedDenKAN

from efficient_kan import KAN as EfficientKAN # 这个是打算去只加在FFN中的KAN，然后我们来看一下他的效果


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class KATEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_groups=8, encoder_kan = None, only_modular_kan = False):
        super(KATEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # --- 修改从这里开始 ---
        self.use_kan = activation.startswith("kan_")
        self.only_modular_kan = only_modular_kan
        # print("only_modular_kan:",self.only_modular_kan)
        # print('arg',only_modular_kan)
        if self.use_kan and self.only_modular_kan == False:
            # if not HAS_KAN:
            #     raise ImportError("KAT_Group is required for KAN activation but not found.")
            mode = activation.replace("kan_", "")  # e.g., "kan_swish" -> "swish"
            # 自动推断设备（假设模型会移到 cuda）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.kan = KAT_Group(num_groups=num_groups, mode=mode, device=device)
            self.activation = None  # 不再使用 F.relu/F.gelu
            # print("using kan")
        else:
            self.activation = F.relu if activation == "relu" else F.gelu
        # --- 修改到这里结束 ---

        # 在中间去加入的一个KAN
        self.encoder_kan = encoder_kan

    def forward(self, x, attn_mask=None, tau=None, delta=None, global_shared_denominator = None):
        # print(x.shape)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # 👉 插入点 B：KAN-2（校准上下文感知的变量表示）
        # Apply KAN-2 if provided
        # print(self.encoder_kan)
        # print(global_shared_denominator)
        if self.encoder_kan is not None and global_shared_denominator is not None:
            # print(x.shape)
            y = self.encoder_kan(x, global_shared_denominator)
            # print("在encoder中使用modular_kan")

        # --- 修改 FFN 部分 ---
        y = y.transpose(-1, 1)  # [B, D, L]
        y = self.conv1(y)       # [B, d_ff, L]

        # print(self.only_modular_kan)
        if self.use_kan and self.only_modular_kan == False:
            # print("现在有使用的是MKAT中的group_kan")
            y = y.transpose(-1, 1)      # [B, L, d_ff]
            y = self.kan(y)             # Apply KAN on [B, L, d_ff]
            # print("在encoder中使用group_kan")
            y = y.transpose(-1, 1)      # back to [B, d_ff, L]
        else:
            # print("现在是没有使用group_kan的情况")
            y = self.activation(y)

        y = self.dropout(y)
        y = self.conv2(y).transpose(-1, 1)  # [B, L, D]
        # --- 修改结束 ---

        return self.norm2(x + y), attn

class KANEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_groups=8, encoder_kan = None, only_modular_kan = False):
        super(KANEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # print('activation',activation)
        self.use_kan = activation.startswith("kan_")
        if activation == "kan_ef":
            self.kan = EfficientKAN([d_ff, d_ff])
        else:
            # 原始 MLP
            self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, x, attn_mask=None, tau=None, delta=None, global_shared_denominator = None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # --- 修改 FFN 部分 ---
        y = y.transpose(-1, 1)  # [B, D, L]
        y = self.conv1(y)       # [B, d_ff, L]

        # print("use_kan",self.use_kan)
        if self.use_kan:
            # print("现在有使用的是在FFN中去使用了KAN")
            B, D_ff, L = y.shape
            y = y.permute(0, 2, 1).contiguous()  # [B, L, d_ff]
            y = y.view(-1, D_ff)                 # [B*L, d_ff]
            y = self.kan(y)                      # [B*L, d_ff]
            y = y.view(B, L, D_ff).permute(0, 2, 1).contiguous()  # [B, d_ff, L]
        else:
            # print("现在是没有使用group_kan的情况")
            y = self.activation(y)

        y = self.dropout(y)
        y = self.conv2(y).transpose(-1, 1)  # [B, L, D]
        # --- 修改结束 ---

        return self.norm2(x + y), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class KATEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(KATEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, global_shared_denominator = None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, global_shared_denominator=global_shared_denominator)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, global_shared_denominator=global_shared_denominator)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, global_shared_denominator=global_shared_denominator)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class KATDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", num_groups=8, decoder_kan=None):
        super(KATDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # --- GroupKAN (from activation string) ---
        self.use_group_kan = activation.startswith("kan_")
        if self.use_group_kan:
            mode = activation.replace("kan_", "")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.group_kan = KAT_Group(num_groups=num_groups, mode=mode, device=device)
            self.activation = None
        else:
            self.activation = F.relu if activation == "relu" else F.gelu

        # --- Modular KAN with shared denominator (MKAT core) ---
        self.decoder_kan = decoder_kan  # Expected to be a SharedDenKAN instance or None

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, global_shared_denominator=None):
        # 1. Self-Attention (masked)
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 2. Cross-Attention (with encoder output)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        y = x = self.norm2(x)

        # 👉 插入点：Modular KAN (shared denominator) —— 对称于 encoder
        if self.decoder_kan is not None and global_shared_denominator is not None:
            y = self.decoder_kan(y, global_shared_denominator)
            # print("In decoder: using modular KAN with shared denominator")

        # 3. FFN / GroupKAN
        y = y.transpose(-1, 1)  # [B, D, L]
        y = self.conv1(y)       # [B, d_ff, L]

        if self.use_group_kan:
            y = y.transpose(-1, 1)      # [B, L, d_ff]
            y = self.group_kan(y)       # Apply GroupKAN
            y = y.transpose(-1, 1)      # [B, d_ff, L]
        else:
            y = self.activation(y)

        y = self.dropout(y)
        y = self.conv2(y).transpose(-1, 1)  # [B, L, D]

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class KATDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(KATDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, global_shared_denominator=None):
        for layer in self.layers:
            x = layer(
                x, cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                tau=tau,
                delta=delta,
                global_shared_denominator=global_shared_denominator  # ← 关键：传给每一层
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x