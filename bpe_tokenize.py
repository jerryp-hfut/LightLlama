import os
import json
import numpy as np
from typing import Dict, List, Tuple, Iterator
from tqdm import tqdm
import re

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.special_token_ids = {t.encode('utf-8'): self.vocab_inv.get(t.encode('utf-8')) for t in self.special_tokens}
        self.merge_lookup = {pair: idx for idx, pair in enumerate(self.merges)}
        self.token_pattern = re.compile(r'\s+|\w+|[^\w\s]', re.UNICODE)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        merges = [(t[0].encode('utf-8'), t[1].encode('utf-8')) for t in merges_json]
        return cls(vocab, merges, special_tokens)

    def encode_single(self, text: str) -> List[int]:
        pre_tokens = self.token_pattern.findall(text)
        tokens = []
        unk_id = self.special_token_ids.get(b'<unk>')
        
        for token_str in pre_tokens:
            if token_str in self.special_tokens:
                special_id = self.special_token_ids.get(token_str.encode('utf-8'))
                if special_id is not None:
                    tokens.append(special_id)
                else:
                    print(f"警告: 特殊 token '{token_str}' 在词汇表中未找到 ID。")
                continue
            chars_byte_list = [bytes([b]) for b in token_str.encode('utf-8')]
            while len(chars_byte_list) > 1:
                min_merge_idx = float('inf')
                best_pos = -1
                for i in range(len(chars_byte_list) - 1):
                    pair = (chars_byte_list[i], chars_byte_list[i + 1])
                    if pair in self.merge_lookup:
                        idx = self.merge_lookup[pair]
                        if idx < min_merge_idx:
                            min_merge_idx = idx
                            best_pos = i
                
                if min_merge_idx == float('inf'):
                    break
                chars_byte_list[best_pos] = chars_byte_list[best_pos] + chars_byte_list[best_pos + 1]
                del chars_byte_list[best_pos + 1]
            for char_seq_bytes in chars_byte_list:
                if char_seq_bytes in self.vocab_inv:
                    tokens.append(self.vocab_inv[char_seq_bytes])
                elif unk_id is not None:
                    tokens.append(unk_id)
                else:
                    raise ValueError(f"Token {char_seq_bytes} (raw: '{token_str.encode('utf-8')}') not in vocabulary and <unk> not set.")
        
        return tokens

    def encode(self, text: str) -> List[int]:
        return self.encode_single(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode_single(text) for text in texts]

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode_single(text)

    def decode(self, ids: List[int]) -> str:
        bytes_seq = b''.join(self.vocab[id] for id in ids if id in self.vocab)
        return bytes_seq.decode('utf-8', errors='replace')


def count_lines(filepath: str) -> int:
    with open(filepath, 'rb') as f:
        count = sum(1 for _ in f)
    return count


def encode_dataset_optimized(input_path: str, output_path: str, tokenizer: Tokenizer, 
                           batch_size: int = 1000, chunk_size: int = 100000):
    
    total_lines = count_lines(input_path)
    print(f"Total lines to process: {total_lines:,}")
    
    end_token_id = tokenizer.special_token_ids.get(b'<|endoftext|>')
    if end_token_id is None:
        raise ValueError("eof not found.")
    
    all_tokens = []
    processed_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(
            total=total_lines, 
            desc=f"Encoding {os.path.basename(input_path)}", 
            unit='lines',
            ncols=100
        )
        
        batch_lines = []
        
        for line in f:
            line = line.strip()
            if line:
                batch_lines.append(line)
            
            if len(batch_lines) >= batch_size:
                for text in batch_lines:
                    tokens = tokenizer.encode_single(text)
                    all_tokens.extend(tokens)
                    all_tokens.append(end_token_id)
                
                processed_lines += len(batch_lines)
                pbar.update(len(batch_lines))
                pbar.set_postfix({
                    'tokens': f"{len(all_tokens):,}",
                    'avg_tokens/line': f"{len(all_tokens)/max(processed_lines,1):.1f}"
                })
                
                batch_lines = []
        if batch_lines:
            for text in batch_lines:
                tokens = tokenizer.encode_single(text)
                all_tokens.extend(tokens)
                all_tokens.append(end_token_id)
            
            processed_lines += len(batch_lines)
            pbar.update(len(batch_lines))
            pbar.set_postfix({
                'tokens': f"{len(all_tokens):,}",
                'avg_tokens/line': f"{len(all_tokens)/processed_lines:.1f}"
            })
        
        pbar.close()
    
    print(f"Converting to numpy array...")
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    print(f"Saving to {output_path}.npy...")
    np.save(output_path, token_array)
    
    print(f"Saved encoded data to {output_path}.npy")
    print(f"Total tokens: {len(token_array):,}")
    print(f"Average tokens per line: {len(token_array)/processed_lines:.1f}")
    print(f"File size: {os.path.getsize(output_path + '.npy') / (1024**2):.1f} MB")


def encode_dataset_streaming(input_path: str, output_path: str, tokenizer: Tokenizer, 
                           batch_size: int = 1000):
    
    total_lines = count_lines(input_path)
    end_token_id = tokenizer.special_token_ids.get(b'<|endoftext|>')
    if end_token_id is None:
        raise ValueError("eof not found. Please ensure '<|endoftext|>' is in special_tokens and trained.")
    all_tokens_list = [] 
    processed_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(
            total=total_lines, 
            desc=f"Streaming {os.path.basename(input_path)}", 
            unit='lines',
            ncols=100,
            miniters=100
        )
        
        batch_lines = []
        
        for line in f:
            line = line.strip()
            if line:
                batch_lines.append(line)
            
            if len(batch_lines) >= batch_size:
                batch_tokens = []
                for text in batch_lines:
                    tokens = tokenizer.encode_single(text)
                    batch_tokens.extend(tokens)
                    batch_tokens.append(end_token_id)
                
                all_tokens_list.extend(batch_tokens)
                processed_lines += len(batch_lines)
                pbar.update(len(batch_lines))
                pbar.set_postfix({
                    'tokens': f"{len(all_tokens_list):,}"
                })
                
                batch_lines = []
        if batch_lines:
            for text in batch_lines:
                tokens = tokenizer.encode_single(text)
                all_tokens_list.extend(tokens)
                all_tokens_list.append(end_token_id)
            processed_lines += len(batch_lines)
            pbar.update(len(batch_lines))
        
        pbar.close()
    print(f"Converting to numpy array...")
    token_array = np.array(all_tokens_list, dtype=np.uint16)
    
    print(f"Saving to {output_path}.npy...")
    np.save(output_path, token_array)
    
    print(f"Streaming encoding completed!")
    print(f"Total tokens: {len(token_array):,}")
    print(f"Average tokens per line: {len(token_array)/processed_lines:.1f}")
    print(f"File size: {os.path.getsize(output_path + '.npy') / (1024**2):.1f} MB")


def main():
    special_tokens = ['<|endoftext|>', '<unk>']
    
    print("🚀 Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        'bpe_params/vocab.json',
        'bpe_params/merges.json',
        special_tokens=special_tokens
    )
    print(f"📚 Vocab size: {len(tokenizer.vocab):,}")
    print(f"🔀 Merge rules: {len(tokenizer.merges):,}")
    use_streaming = True
    
    if use_streaming:
        print("🌊 Using streaming mode (memory-friendly)")
        encode_dataset_streaming(
            'data/TinyStoriesV2-GPT4-train.txt',
            'data/TinyStoriesV2-GPT4-train', 
            tokenizer,
            batch_size=2000
        )
        encode_dataset_streaming(
            'data/TinyStoriesV2-GPT4-valid.txt',
            'data/TinyStoriesV2-GPT4-valid',
            tokenizer,
            batch_size=2000
        )
    else:
        print("⚡ Using optimized batch mode")
        encode_dataset_optimized(
            'data/TinyStoriesV2-GPT4-train.txt',
            'data/TinyStoriesV2-GPT4-train',
            tokenizer,
            batch_size=1000
        )
        encode_dataset_optimized(
            'data/TinyStoriesV2-GPT4-valid.txt',
            'data/TinyStoriesV2-GPT4-valid',
            tokenizer,
            batch_size=1000
        )
    
    print("completed!")


if __name__ == "__main__":
    main()