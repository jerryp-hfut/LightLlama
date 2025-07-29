import re
import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def bpe_trainer(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = {}
    merges: List[Tuple[bytes, bytes]] = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"'{input_path}' path not found.")
        return {}, []
    except Exception as e:
        print(f"error reading file: {e}")
        return {}, []
    tokens_from_text = re.findall(r'\s+|\w+|[^\w\s]', text, re.UNICODE)
    word_freqs = Counter(tokens_from_text)
    word_splits = {}
    for token, freq in word_freqs.items():
        word_splits[tuple([token.encode('utf-8')])] = freq

    vocab_id = 0
    for token in special_tokens:
        vocab[vocab_id] = token.encode('utf-8')
        vocab_id += 1
    alphabet = set()
    for chars_tuple in word_splits:
        for char_bytes in chars_tuple:
            if len(char_bytes) == 1:
                alphabet.add(char_bytes)
            else:
                for b in char_bytes:
                    alphabet.add(bytes([b]))
    for c in text.encode('utf-8'):
        alphabet.add(bytes([c]))

    for char_bytes in sorted(alphabet):
        if char_bytes not in vocab.values():
            vocab[vocab_id] = char_bytes
            vocab_id += 1

    print(f"original vocab size: {len(vocab)}")
    print(f"target vocab size: {vocab_size}")
    num_merges = vocab_size - len(vocab)
    new_word_splits_for_bpe = {}
    for token_bytes, freq in word_splits.items():
        new_word_splits_for_bpe[tuple(bytes([b]) for b in token_bytes[0])] = freq
    word_splits = new_word_splits_for_bpe


    for i in range(num_merges):
        pairs = defaultdict(int)

        for chars, freq in word_splits.items():
            for j in range(len(chars) - 1):
                pairs[(chars[j], chars[j + 1])] += freq
        if not pairs:
            print(f"No more can be merged. Current size of vocab is: {i}")
            break
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        new_word_splits = {}
        for chars, freq in word_splits.items():
            new_chars = []
            j = 0
            while j < len(chars):
                if j < len(chars) - 1 and (chars[j], chars[j + 1]) == best_pair:
                    new_chars.append(chars[j] + chars[j + 1])
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1
            new_word_splits[tuple(new_chars)] = freq
        word_splits = new_word_splits
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[vocab_id] = new_token
        vocab_id += 1

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{num_merges} merges. current vocab size: {len(vocab)}")

    print(f"training complete. final vocab size: {len(vocab)}")
    print(f"num of merges: {len(merges)}")
    return vocab, merges

def save_bpe_model_json(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], save_dir: str = "bpe_params"):
    os.makedirs(save_dir, exist_ok=True)

    vocab_for_json = {str(k): v.decode('utf-8', errors='ignore') for k, v in vocab.items()}

    merges_for_json = [(t[0].decode('utf-8', errors='ignore'), t[1].decode('utf-8', errors='ignore')) for t in merges]

    vocab_path = os.path.join(save_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_for_json, f, ensure_ascii=False, indent=2)

    merges_path = os.path.join(save_dir, "merges.json")
    with open(merges_path, 'w', encoding='utf-8') as f:
        json.dump(merges_for_json, f, ensure_ascii=False, indent=2)

    info = {
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "format": "json"
    }
    info_path = os.path.join(save_dir, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"vocab and merges saved to: {save_dir}/")

def main():
    special_tokens = ['<|endoftext|>', '<unk>'] 
    vocab, merges = bpe_trainer('data/TinyStoriesV2-GPT4-valid.txt', 10000, special_tokens)
    save_bpe_model_json(vocab, merges)

if __name__ == "__main__":
    main()