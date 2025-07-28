import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        theta = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        positions = torch.arange(max_seq_len).unsqueeze(1)
        angles = positions * theta
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, start_pos=0):
        batch, seq_len, _ = x.shape
        cos = self.cos[start_pos:start_pos + seq_len, :]
        sin = self.sin[start_pos:start_pos + seq_len, :]
        cos = cos.unsqueeze(0).expand(batch, -1, -1)
        sin = sin.unsqueeze(0).expand(batch, -1, -1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

class Multiheadselfattention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, max_seq_len=2048, theta=10000, is_causal=True):
        super(Multiheadselfattention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_causal = is_causal

    def forward(self, x, start_pos=0, cache=None):
        batch, seq_len, d_model = x.shape
        head_dim = d_model // self.num_heads
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = self.rope(Q, start_pos)
        K = self.rope(K, start_pos)
        q = Q.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = K.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = V.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            k_cache, v_cache = k, v
        new_cache = (k, v)
        attn = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(float(head_dim), device=x.device))
        if self.is_causal:
            mask = torch.triu(torch.ones(seq_len, k.size(2), device=x.device), diagonal=k.size(2) - seq_len + 1) * float('-inf')
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn = attn + mask
        attn_norm = torch.softmax(attn, dim=-1)
        attn_norm = self.dropout(attn_norm)
        attn_out = (attn_norm @ v).transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.w_o(attn_out)
        return output, new_cache

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.gamma

class Swiglu(nn.Module):
    def __init__(self, d_model):
        super(Swiglu, self).__init__()
        # d_ff = d_model * 8 // 3
        d_ff = 1344
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

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
        normout1 = self.norm1(x)
        attn_out, new_cache = self.mhsa(normout1, start_pos, cache)
        stage1 = attn_out + x
        normout2 = self.norm2(stage1)
        ffn_out = self.ffn(normout2)
        stage2 = ffn_out + stage1
        return stage2, new_cache

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
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