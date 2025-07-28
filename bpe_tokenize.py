import json
import numpy as np
from typing import Dict, List, Tuple, Iterator
from tqdm import tqdm

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.special_token_ids = {t.encode('utf-8'): self.vocab_inv.get(t.encode('utf-8')) for t in self.special_tokens}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        merges = [(t[0].encode('utf-8'), t[1].encode('utf-8')) for t in merges_json]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        import re
        pre_tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        tokens = []
        unk_id = self.special_token_ids.get(b'<unk>')
        for token in pre_tokens:
            if token in self.special_tokens:
                tokens.append(self.special_token_ids[token.encode('utf-8')])
                continue
            chars = list(token.encode('utf-8'))
            while len(chars) > 1:
                min_merge_idx = float('inf')
                merge_pair = None
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    idx = next((j for j, m in enumerate(self.merges) if m == pair), float('inf'))
                    if idx < min_merge_idx:
                        min_merge_idx = idx
                        merge_pair = pair
                if min_merge_idx == float('inf'):
                    break
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == merge_pair:
                        new_chars.append(chars[i] + chars[i + 1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            for char in chars:
                if char in self.vocab_inv:
                    tokens.append(self.vocab_inv[char])
                elif unk_id is not None:
                    tokens.append(unk_id)
                else:
                    raise ValueError(f"Token {char} not in vocabulary and <unk> not set")
        return tokens

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        bytes_seq = b''.join(self.vocab[id] for id in ids if id in self.vocab)
        return bytes_seq.decode('utf-8', errors='replace')

def encode_dataset(input_path: str, output_path: str, tokenizer: Tokenizer):
    """将文本数据集编码为 .npy 文件"""
    with open(input_path, 'r', encoding='utf-8') as f:
        token_ids = []
        for line in tqdm(f, desc=f"Encoding {input_path}"):
            if line.strip():
                token_ids.extend(tokenizer.encode(line))
                token_ids.append(tokenizer.special_token_ids[b'<|endoftext|>'])
    
    token_array = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, token_array)
    print(f"Saved encoded data to {output_path}, total tokens: {len(token_array)}")

def main():
    special_tokens = ['<|endoftext|>', '<unk>']
    tokenizer = Tokenizer.from_files(
        'bpe_params/vocab.json',
        'bpe_params/merges.json',
        special_tokens=special_tokens
    )

    encode_dataset(
        'data/TinyStoriesV2-GPT4-train.txt',
        'data/TinyStoriesV2-GPT4-train.npy',
        tokenizer
    )
    encode_dataset(
        'data/TinyStoriesV2-GPT4-valid.txt',
        'data/TinyStoriesV2-GPT4-valid.npy',
        tokenizer
    )

if __name__ == "__main__":
    main()