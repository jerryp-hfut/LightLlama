import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        theta = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        positions = torch.arange(max_seq_len).unsqueeze(1)
        angles = positions * theta
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))
    def forward(self, x, start_pos=0):
        batch, seq_len, d_model = x.shape
        end_pos = min(start_pos + seq_len, self.max_seq_len)
        actual_seq_len = end_pos - start_pos
        if actual_seq_len <= 0:
            return x
        cos = self.cos[start_pos:end_pos, :]
        sin = self.sin[start_pos:end_pos, :]
        if cos.size(0) != seq_len:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        cos = cos.unsqueeze(0).expand(batch, -1, -1)
        sin = sin.unsqueeze(0).expand(batch, -1, -1)
        x1 = x[..., 0::2]  # 偶
        x2 = x[..., 1::2]  # 奇
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        x_rot = torch.stack([x_rot1, x_rot2], dim=-1)
        x_rot = x_rot.flatten(-2)
        return x_rot

class Multiheadselfattention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, max_seq_len=2048, theta=10000, is_causal=True):
        super(Multiheadselfattention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, theta)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_causal = is_causal

    def forward(self, x, start_pos=0, cache=None):
        batch, seq_len, d_model = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_rot = torch.zeros_like(q)
        k_rot = torch.zeros_like(k)
        for head in range(self.num_heads):
            q_rot[:, head, :, :] = self.rope(q[:, head, :, :], start_pos)
            k_rot[:, head, :, :] = self.rope(k[:, head, :, :], start_pos)
        if cache is not None:
            k_cache, v_cache = cache
            k_rot = torch.cat([k_cache, k_rot], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k_rot, v)
        attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        if self.is_causal:
            seq_len_k = k_rot.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len_k, device=x.device, dtype=torch.bool),
                diagonal=seq_len_k - seq_len + 1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(attn, dim=-1)
        if torch.isnan(attn_weights).any():
            print("NaN detected in attention weights!")
            print(f"Attention scores range: [{attn.min():.6f}, {attn.max():.6f}]")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.w_o(attn_out)
        return output, new_cache

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        if torch.isnan(rms).any() or torch.isinf(rms).any():
            print("NaN/Inf detected in RMSNorm!")
            rms = torch.clamp(rms, min=self.eps)
        
        normalized = x / rms
        return normalized * self.gamma

class Swiglu(nn.Module):
    def __init__(self, d_model):
        super(Swiglu, self).__init__()
        self.d_ff = int(d_model * 8 / 3)
        self.d_ff = ((self.d_ff + 7) // 8) * 8
        
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False)

    def forward(self, x):
        w1x = self.w1(x)
        silu_out = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x)
        temp = silu_out * w3x
        output = self.w2(temp)
        
        return output

class Transformerblock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, max_seq_len=2048, theta=10000, is_causal=True):
        super(Transformerblock, self).__init__()
        self.mhsa = Multiheadselfattention(d_model, num_heads, dropout_rate, max_seq_len, theta, is_causal)
        self.ffn = Swiglu(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, start_pos=0, cache=None):
        norm1_out = self.norm1(x)
        attn_out, new_cache = self.mhsa(norm1_out, start_pos, cache)
        x = x + attn_out
        norm2_out = self.norm2(x)
        ffn_out = self.ffn(norm2_out)
        x = x + ffn_out
        
        return x, new_cache

class LLaMA(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout_rate=0.1, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            Transformerblock(d_model, num_heads, dropout_rate, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Xavier init
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.gamma)

    def forward(self, x, start_pos=0, cache=None):
        if cache is None:
            cache = [None] * len(self.layers)
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, start_pos, cache[i])
            new_cache.append(layer_cache)
        x = self.norm(x)
        return x, new_cache

def count_parameters(model, unit='B'):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_billions = total_params / 1e9
    return params_in_billions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch = 4
    seq_len = 256
    d_model = 512
    num_heads = 16
    num_layers = 4
    input = torch.randn(batch, seq_len, d_model, device=device)
    print("Input shape:", input.shape)

    model = LLaMA(d_model, num_heads, num_layers, max_seq_len=2048).to(device)
    params_in_billions = count_parameters(model)
    print(f"Model params: {params_in_billions:.3f} B")

    output, cache = model(input)
    print("Output shape:", output.shape)

if __name__ == '__main__':
    main()