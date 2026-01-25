import json
from transformers import AutoTokenizer
import numpy as np
import argparse
import os  # Add this import

def read_vocab_from_file(tok_config_path):
    with open(tok_config_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        vocab = data["model"]["vocab"]
    return vocab

def read_vocab(tok_path):
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    # Try .vocab attribute first, fall back to get_vocab() method if needed
    if hasattr(tok, 'vocab') and tok.vocab is not None:
        return tok.vocab
    elif hasattr(tok, 'get_vocab'):
        return tok.get_vocab()
    else:
        # Fallback: try to get vocab from tokenizer's underlying tokenizer
        if hasattr(tok, 'tokenizer') and hasattr(tok.tokenizer, 'get_vocab'):
            return tok.tokenizer.get_vocab()
        else:
            raise AttributeError(f"Tokenizer from {tok_path} does not have vocab attribute or get_vocab() method")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-tokenizer-path", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("-t", "--target-tokenizer-path", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("-o", "--output-path", type=str, default="./data/Vocab_count/qwen2-7b2pythia.json")
    
    args = parser.parse_args()

    vocab_overlap = {}

    # new vocab
    model1_vocab = read_vocab(
        tok_path=args.target_tokenizer_path
    )

    # old vocab
    model2_vocab = read_vocab(
        tok_path=args.source_tokenizer_path
    )

    save_path = args.output_path

    num_vocab = 0
    num_overlap = 0
    model1_vocab = dict(sorted(model1_vocab.items(), key=lambda item: item[1], reverse=False))
    for key, value in model1_vocab.items():
        num_vocab += 1
        if key in model2_vocab:
            row = model1_vocab[key]
            col = model2_vocab[key]
            vocab_overlap[row] = col
            num_overlap += 1

    # print(vocab_overlap)
    print(f"{num_overlap}/{num_vocab} ({num_overlap*100/num_vocab:.2f}%)")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(vocab_overlap, file, indent="\t", ensure_ascii=False)
