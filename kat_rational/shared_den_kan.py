# shared_den_kan.py
import torch
import torch.nn as nn
import os
import json

# shared_den_kan.py :: rational_full
def rational_full(x, numerator, denominator):
    *dims, C = x.shape
    # x = x.view(-1, C)  # [N, C]
    # print("x:",x.shape)
    x = x.reshape(-1, C)
    #print("x:",x.shape)
    N = x.size(0)

    K = numerator.size(1)
    Q = denominator.size(1)
    max_deg = max(K, Q + 1)

    # Build powers: [N, C, max_deg]
    powers = []

    # x^0 = 1 → [N, C, 1]
    powers.append(torch.ones(N, C, 1, device=x.device, dtype=x.dtype))

    if max_deg > 1:
        # x^1 = x → [N, C, 1]
        powers.append(x.unsqueeze(-1))
        xp = x
        for d in range(2, max_deg):
            xp = xp * x  # [N, C]
            powers.append(xp.unsqueeze(-1))  # [N, C, 1]

    powers = torch.cat(powers, dim=-1)  # [N, C, max_deg] ✅ all 3D

    # Numerator: sum_{k=0}^{K-1} a_k x^k
    num_out = (powers[:, :, :K] * numerator.unsqueeze(0)).sum(-1)  # [N, C]

    # Denominator: 1 + sum_{q=1}^{Q} |b_q| |x|^q
    if Q > 0:
        den_powers = powers[:, :, 1:1+Q]  # [N, C, Q]
        den_weights = denominator.unsqueeze(0).abs()  # [1, C, Q]
        # print("den_powers:",den_powers.shape)
        # print("den_weigths",den_weights.shape)
        den_sum = (den_powers * den_weights).sum(-1)  # [N, C]
        den_out = 1.0 + den_sum
    else:
        den_out = torch.ones_like(num_out)

    # print("num:",num_out.shape)
    # print("den_out",den_out.shape)
    y = num_out / den_out
    # print(y.shape)
    return y.view(*dims, C)


class SharedDenKAN(nn.Module):
    """
    A KAN layer that uses an external shared denominator.
    Used as: y = rational(x, self.numerator, shared_denominator)
    """
    def __init__(self, in_features, init_mode="gelu"):
        super().__init__()
        self.in_features = in_features

        # Load init
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as f:
                init_data = json.load(f)
            w_num = torch.tensor(init_data[init_mode]["init_w_numerator"])  # [K]
            w_den = torch.tensor(init_data[init_mode]["init_w_denominator"])  # [Q]
        except:
            w_num = torch.randn(6)
            w_den = torch.randn(4)

        # Only numerator is owned by this module
        self.numerator = nn.Parameter(w_num.repeat(in_features, 1).float())
        # Denominator will be passed in forward()

    def forward(self, x, shared_denominator):
        """
        x: [..., C]
        shared_denominator: [C, Q] ← same tensor used by other KANs
        """
        return rational_full(x, self.numerator, shared_denominator)