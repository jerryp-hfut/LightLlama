import torch
import numpy as np
from model.llama import LLaMA, RMSNorm
from pre_train import LanguageModel
import argparse
import json
from bpe_tokenize import Tokenizer

def load_tokenizer():
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer.from_files(
        'bpe_params/vocab.json',
        'bpe_params/merges.json',
        special_tokens=special_tokens
    )
    return tokenizer

def load_model(checkpoint_path, device, context_length=512):
    vocab_size = 15018
    d_model = 768
    num_heads = 16
    num_layers = 16
    model = LanguageModel(vocab_size, d_model, num_heads, num_layers, max_seq_len=context_length)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def sample_next_token(logits, temperature=1.0, top_p=0.0, top_k=0, repetition_penalty=1.0, generated=None):
    logits = logits / temperature

    # repitition penalty
    if repetition_penalty > 1.0 and generated is not None:
        for token_id in set(generated[-10:]):
            logits[token_id] /= repetition_penalty

    if top_p > 0:
        # Top-p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    elif top_k > 0:
        # Top-k
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return indices[idx]
    else:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

def generate(model, tokenizer, prompt, device, max_new_tokens=128, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, context_length=512):
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[:context_length]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids.copy()
    cache = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model(input_tensor, start_pos=0, cache=cache)
            logits = logits[0, -1, :]
            next_token = sample_next_token(logits, temperature, top_p, top_k, repetition_penalty, generated)
            next_token_id = next_token.item() if isinstance(next_token, torch.Tensor) else int(next_token)
        generated.append(next_token_id)
        if next_token_id == tokenizer.special_token_ids.get(b'<|endoftext|>'):
            break
        input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description="LLaMA inference script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling (set to 0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling (set to 0 to disable)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty (1.0 means no penalty)')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    model = load_model(args.checkpoint, device, context_length=args.context_length)

    print("\nPrompt:", args.prompt)
    output = generate(
        model, tokenizer, args.prompt, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        context_length=args.context_length
    )
    print("\nGenerated text:")
    print(output)

if __name__ == "__main__":
    main()