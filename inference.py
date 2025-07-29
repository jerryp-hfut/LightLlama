import torch
import numpy as np
from model.llama import LLaMA, RMSNorm
from pre_train import LanguageModel
import argparse

from bpe_tokenize import Tokenizer

def load_tokenizer():
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer.from_files(
        'bpe_params/vocab.json',
        'bpe_params/merges.json',
        special_tokens=special_tokens
    )
    return tokenizer

def load_model(checkpoint_path, device, context_length=128):
    vocab_size = 10000
    d_model = 256
    num_heads = 8
    num_layers = 2
    model = LanguageModel(vocab_size, d_model, num_heads, num_layers, max_seq_len=context_length)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def sample_next_token(logits, temperature=1.0, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return indices[idx]
    else:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

def generate(model, tokenizer, prompt, device, max_new_tokens=100, temperature=1.0, top_k=20, context_length=128):
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[:context_length]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids.copy()
    cache = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model(input_tensor, start_pos=0, cache=cache)
            logits = logits[0, -1, :]
            next_token = sample_next_token(logits, temperature, top_k)
            next_token_id = next_token.item() if isinstance(next_token, torch.Tensor) else int(next_token)
        generated.append(next_token_id)
        if next_token_id == tokenizer.special_token_ids.get(b'<|endoftext|>'):
            break
        # 滑动窗口
        input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description="LLaMA inference script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k sampling')
    parser.add_argument('--context_length', type=int, default=128, help='Context length')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = load_tokenizer()
    model = load_model(args.checkpoint, device, context_length=args.context_length)
    print("\nPrompt:", args.prompt)
    output = generate(
        model, tokenizer, args.prompt, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        context_length=args.context_length
    )
    print("\nGenerated text:")
    print(output)

if __name__ == "__main__":
    main()
