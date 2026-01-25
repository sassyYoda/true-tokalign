import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import argparse

_EMBED_DICT = {
    "gpt_neox": "gpt_neox.embed_in.weight",
    "llama": "model.embed_tokens.weight",
    "mistral": "model.embed_tokens.weight",
}

_LMHEAD_DICT = {
    "gpt_neox": "embed_out.weight",
    "llama": "lm_head.weight",
    "mistral": "lm_head.weight",
}

def trans2switch(
    trans_path="./log/qwen2-7b2pythia/glove.json",
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2qwen2-7b",
    tgt_tok_path="./data/qwen2-7b",
    random_shuffle=-1,
):
    import sys
    sys.stdout.flush()
    
    print("Step 1: Loading source model...", flush=True)
    try:
        src_model = AutoModelForCausalLM.from_pretrained(
            src_clm_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        print(f"✓ Model loaded. Model type: {src_model.config.model_type}", flush=True)
        print(f"  Device: {next(src_model.parameters()).device}", flush=True)
    except Exception as e:
        print(f"✗ Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print("Step 2: Loading target tokenizer...", flush=True)
    try:
        tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded. Vocab size: {len(tgt_tok)}", flush=True)
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}", flush=True)
        raise

    print("Step 3: Loading transition matrix...", flush=True)
    try:
        with open(trans_path, "r") as f:
            trans = json.load(f)
        print(f"✓ Transition matrix loaded: {len(trans)} entries", flush=True)
    except Exception as e:
        print(f"✗ Error loading transition matrix: {e}", flush=True)
        raise
    
    print("Step 4: Accessing model parameters...", flush=True)
    try:
        src_params = dict(src_model.named_parameters())
        print(f"✓ Model parameters accessed. Total params: {len(src_params)}", flush=True)
    except Exception as e:
        print(f"✗ Error accessing parameters: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 5: Getting embedding layers...", flush=True)
    try:
        model_type = src_model.config.model_type
        print(f"  Model type: {model_type}", flush=True)
        
        if model_type not in _EMBED_DICT:
            raise ValueError(f"Model type {model_type} not supported. Supported types: {list(_EMBED_DICT.keys())}")
        
        embed_key = _EMBED_DICT[model_type]
        lmhead_key = _LMHEAD_DICT[model_type]
        print(f"  Embed key: {embed_key}", flush=True)
        print(f"  LM head key: {lmhead_key}", flush=True)
        
        src_embed = src_params[embed_key]
        src_lm_head = src_params[lmhead_key]
        print(f"✓ Embedding shape: {src_embed.shape}, LM head shape: {src_lm_head.shape}", flush=True)
        print(f"  Embed device: {src_embed.device}, LM head device: {src_lm_head.device}", flush=True)
    except Exception as e:
        print(f"✗ Error getting embeddings: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 6: Validating shapes...", flush=True)
    assert src_embed.shape[0] == src_lm_head.shape[0]

    hid_dim = src_embed.shape[1]
    src_len = src_embed.shape[0]
    tgt_len = len(list(trans.keys()))
    
    print(f"  Source vocab size: {src_len}, Target vocab size: {tgt_len}, Hidden dim: {hid_dim}", flush=True)

    print("Step 7: Creating target embeddings...", flush=True)
    try:
        tgt_embed = torch.zeros((tgt_len, hid_dim), device=src_embed.device, dtype=src_embed.dtype)
        tgt_lm_head = torch.zeros((tgt_len, hid_dim), device=src_lm_head.device, dtype=src_lm_head.dtype)
        print(f"✓ Target embeddings created on device: {tgt_embed.device}", flush=True)
    except Exception as e:
        print(f"✗ Error creating target embeddings: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 8: Copying embeddings...", flush=True)
    try:
        for i in range(tgt_len):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{tgt_len}", flush=True)
            tj = trans[f"{i}"]
            # random_shuffle experiment
            if random_shuffle > 0 and random.random() < random_shuffle:
                tj = random.randint(0, src_len-1)

            tgt_embed[i] = src_embed[tj]
            tgt_lm_head[i] = src_lm_head[tj]
        print(f"✓ Embeddings copied", flush=True)
    except Exception as e:
        print(f"✗ Error copying embeddings: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 9: Resizing token embeddings...", flush=True)
    try:
        src_model.resize_token_embeddings(tgt_len)
        print(f"✓ Token embeddings resized to {tgt_len}", flush=True)
    except Exception as e:
        print(f"✗ Error resizing embeddings: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 10: Updating parameters...", flush=True)
    try:
        src_params[_EMBED_DICT[src_model.config.model_type]] = tgt_embed.to(torch.bfloat16)
        src_params[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(torch.bfloat16)
        print(f"✓ Parameters updated", flush=True)
    except Exception as e:
        print(f"✗ Error updating parameters: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("Step 11: Loading state dict...", flush=True)
    try:
        src_model.load_state_dict(src_params)
        print(f"✓ State dict loaded", flush=True)
    except Exception as e:
        print(f"✗ Error loading state dict: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print("Step 12: Saving model...", flush=True)
    try:
        src_model.save_pretrained(tgt_clm_path)
        print(f"✓ Model saved to {tgt_clm_path}", flush=True)
    except Exception as e:
        print(f"✗ Error saving model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print("Step 13: Saving tokenizer...", flush=True)
    try:
        tgt_tok.save_pretrained(tgt_clm_path)
        print(f"✓ Tokenizer saved to {tgt_clm_path}", flush=True)
    except Exception as e:
        print(f"✗ Error saving tokenizer: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print("✓ All steps completed successfully!", flush=True)

def random_permute(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2qwen2-7b",
    tgt_tok_path="./data/qwen2-7b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(src_clm_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)

    src_params = dict(src_model.named_parameters())

    src_embed = src_params[_EMBED_DICT[src_model.config.model_type]]
    src_lm_head = src_params[_LMHEAD_DICT[src_model.config.model_type]]

    assert src_embed.shape[0] == src_lm_head.shape[0]

    src_len, hid_dim = src_embed.shape[0], src_embed.shape[1]

    tgt_len = len(tgt_tok)

    tgt_embed = torch.zeros((tgt_len, hid_dim))
    tgt_lm_head = torch.zeros((tgt_len, hid_dim))

    #### Method 1: Re-arrange matrix
    for i in range(tgt_len):
        tj = random.randint(0, src_len-1)

        tgt_embed[i] = src_embed[tj]
        tgt_lm_head[i] = src_lm_head[tj]

    src_model.resize_token_embeddings(len(tgt_tok))

    src_params[_EMBED_DICT[src_model.config.model_type]] = tgt_embed.to(torch.bfloat16)
    src_params[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(torch.bfloat16)

    src_model.load_state_dict(src_params)
    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)

def random_initial_all(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2qwen2-7b",
    tgt_tok_path="./data/qwen2-7b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(src_clm_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)
    
    src_params = dict(src_model.named_parameters())

    src_embed = src_params[_EMBED_DICT[src_model.config.model_type]]
    src_lm_head = src_params[_LMHEAD_DICT[src_model.config.model_type]]

    assert src_embed.shape[0] == src_lm_head.shape[0]

    src_len, hid_dim = src_embed.shape[0], src_embed.shape[1]

    tgt_len = len(tgt_tok)

    src_model.resize_token_embeddings(src_len + tgt_len)

    resized_params = dict(src_model.named_parameters())

    src_params[_EMBED_DICT[src_model.config.model_type]] = resized_params[_EMBED_DICT[src_model.config.model_type]][src_len:]
    src_params[_LMHEAD_DICT[src_model.config.model_type]] = resized_params[_LMHEAD_DICT[src_model.config.model_type]][src_len:]

    src_model.resize_token_embeddings(tgt_len)

    src_model.load_state_dict(src_params)
    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)


def random_initial_aug(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2qwen2-7b",
    tgt_tok_path="./data/qwen2-7b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(src_clm_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)

    src_model.resize_token_embeddings(len(tgt_tok))

    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--one2one-matrix-path", type=str, default="./data/pythia2qwen2-7b/glove.json")
    parser.add_argument("-s", "--source-model-path", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("-t", "--target-tokenizer-path", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("-o", "--output-model-path", type=str, default="./data/pythia2qwen2-7b/glove")
    parser.add_argument("-r", "--random-shuffle-percentage", type=float, default=-1, help="The percentage of token pairs that are randomly shuffled rather than map to the target.")

    args = parser.parse_args()

    trans2switch(
        trans_path=args.one2one_matrix_path,
        src_clm_path=args.source_model_path,
        tgt_clm_path=args.output_model_path,
        tgt_tok_path=args.target_tokenizer_path,
        random_shuffle=args.random_shuffle_percentage
    )
