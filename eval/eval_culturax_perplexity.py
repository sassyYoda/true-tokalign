import argparse
import json
import os
from typing import Dict, List, Optional
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

LANGUAGE_RESOURCES = {
    "high": ["ar", "de", "en", "ja", "zh"],
    "medium": ["bn", "ko", "th", "uk", "vi"],
    "low": ["ta", "te", "ur"]
}

ALL_LANGUAGES = [lang for langs in LANGUAGE_RESOURCES.values() for lang in langs]

PYTHIA_BASE_MODEL = "EleutherAI/pythia-1b"

LANGUAGE_TO_RESOURCE = {lang: level for level, langs in LANGUAGE_RESOURCES.items() for lang in langs}


def _setup_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_pythia_tokenizer(cache_dir: Optional[str] = None):
    print(f"Loading Pythia tokenizer from {PYTHIA_BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        PYTHIA_BASE_MODEL,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    return _setup_tokenizer(tokenizer)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
):
    print(f"Loading model from: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    _setup_tokenizer(tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def _extract_texts(dataset, language, max_samples, text_field="text"):
    texts = []
    processed = 0
    
    for example in tqdm(dataset, desc=f"Streaming {language} texts", total=max_samples if max_samples else None):
        processed += 1
        
        lang_field = example.get('language') or example.get('lang')
        if lang_field and lang_field.lower() != language.lower():
            continue
        
        text = example.get(text_field, "")
        if text and isinstance(text, str) and len(text.strip()) > 50:
            texts.append(text.strip())
        
        if max_samples and len(texts) >= max_samples:
            break
    
    return texts, processed


def load_culturax_language(
    language: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    dataset_name: str = "cultura_x"
) -> List[str]:
    print(f"\nLoading CulturaX {language} ({split})...")
    print(f"Using streaming mode to load only needed samples (max_samples={max_samples})...")
    
    for use_lang_config in [True, False]:
        try:
            if use_lang_config:
                dataset = load_dataset(
                    dataset_name, language, split=split,
                    cache_dir=cache_dir, trust_remote_code=True, streaming=True
                )
                texts, processed = _extract_texts(dataset, language, max_samples)
                print(f"Extracted {len(texts)} valid {language} texts from {processed} processed examples (streaming mode)")
            else:
                dataset = load_dataset(
                    dataset_name, split=split,
                    cache_dir=cache_dir, trust_remote_code=True, streaming=True
                )
                texts, _ = _extract_texts(dataset, language, max_samples)
                print(f"Extracted {len(texts)} valid {language} texts (alternative streaming method)")
            return texts
        except Exception as e:
            if use_lang_config:
                print(f"Error loading CulturaX {language}: {e}")
                print("Trying alternative loading method with streaming...")
            else:
                print(f"Failed to load CulturaX {language} with streaming: {e}")
                print("Note: CulturaX may download metadata/index files on first run.")
                print("Subsequent runs will be faster as files are cached.")
    
    return []


def compute_normalized_perplexity(
    model,
    model_tokenizer,
    pythia_tokenizer,
    texts: List[str],
    language: str,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    stride: Optional[int] = None
) -> Dict:
    print(f"\nComputing normalized perplexity for {language}...")
    print(f"Batch size: {batch_size}, Max length: {max_length}")
    
    if stride is None:
        stride = max_length
    
    model.eval()
    total_loss = 0.0
    total_pythia_tokens = 0
    num_valid_samples = 0
    
    for batch_idx in tqdm(range(0, len(texts), batch_size), desc=f"Processing {language}"):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_losses = []
        batch_pythia_token_counts = []
        
        for text in batch_texts:
            try:
                pythia_encodings = pythia_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=None,
                    add_special_tokens=True
                )
                pythia_input_ids = pythia_encodings["input_ids"]
                
                pythia_seq_len = pythia_input_ids.shape[1]
                pythia_num_tokens = pythia_seq_len - 1
                if pythia_num_tokens <= 0:
                    continue
                
                normalized_text = pythia_tokenizer.decode(pythia_input_ids[0], skip_special_tokens=True)
                
                model_encodings = model_tokenizer(
                    normalized_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=None,
                    add_special_tokens=True
                )
                model_input_ids = model_encodings["input_ids"].to(device)
                
                seq_len = model_input_ids.shape[1]
                if seq_len < 2:
                    continue
                
                if seq_len <= max_length:
                    with torch.no_grad():
                        outputs = model(input_ids=model_input_ids, labels=model_input_ids)
                        model_num_tokens = seq_len - 1
                        if model_num_tokens > 0:
                            batch_losses.append(outputs.loss.item() * model_num_tokens)
                            batch_pythia_token_counts.append(pythia_num_tokens)
                            num_valid_samples += 1
                else:
                    window_losses = []
                    with torch.no_grad():
                        for start_idx in range(0, seq_len - max_length + 1, stride):
                            end_idx = min(start_idx + max_length, seq_len)
                            window_ids = model_input_ids[:, start_idx:end_idx]
                            if window_ids.shape[1] >= 2:
                                outputs = model(input_ids=window_ids, labels=window_ids)
                                window_losses.append(outputs.loss.item() * (window_ids.shape[1] - 1))
                    if window_losses:
                        batch_losses.append(sum(window_losses))
                        batch_pythia_token_counts.append(pythia_num_tokens)
                        num_valid_samples += 1
                            
            except Exception as e:
                print(f"Error processing {language} sample {batch_idx}: {e}")
                continue
        
        if batch_losses:
            for loss, pythia_tokens in zip(batch_losses, batch_pythia_token_counts):
                total_loss += loss
                total_pythia_tokens += pythia_tokens
    
    if total_pythia_tokens > 0:
        avg_loss = total_loss / total_pythia_tokens
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    results = {
        "language": language,
        "num_samples": num_valid_samples,
        "total_pythia_tokens": total_pythia_tokens,
        "average_loss": avg_loss,
        "perplexity": perplexity
    }
    
    print(f"{language}: Perplexity = {perplexity:.4f} (loss: {avg_loss:.4f}, {num_valid_samples} samples, {total_pythia_tokens} Pythia tokens)")
    
    return results


def evaluate_all_languages(
    model_path: str,
    languages: List[str] = None,
    split: str = "train",
    max_samples_per_lang: Optional[int] = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    dataset_name: str = "cultura_x"
) -> Dict:
    if languages is None:
        languages = ALL_LANGUAGES
    
    pythia_tokenizer = load_pythia_tokenizer(cache_dir=cache_dir)
    
    model, model_tokenizer = load_model_and_tokenizer(model_path, device=device, cache_dir=cache_dir)
    
    all_results = {}
    results_by_resource = {
        "high": {},
        "medium": {},
        "low": {}
    }
    
    for language in languages:
        resource_level = LANGUAGE_TO_RESOURCE.get(language)
        if resource_level is None:
            print(f"Warning: Language {language} not in resource level mapping, skipping")
            continue
        
        texts = load_culturax_language(
            language=language,
            split=split,
            max_samples=max_samples_per_lang,
            cache_dir=cache_dir,
            dataset_name=dataset_name
        )
        
        if not texts:
            print(f"Warning: No texts loaded for {language}, skipping")
            continue
        
        results = compute_normalized_perplexity(
            model=model,
            model_tokenizer=model_tokenizer,
            pythia_tokenizer=pythia_tokenizer,
            texts=texts,
            language=language,
            batch_size=batch_size,
            max_length=max_length,
            device=device
        )
        
        all_results[language] = results
        results_by_resource[resource_level][language] = results
    
    for level in ["high", "medium", "low"]:
        if results_by_resource[level]:
            perplexities = [r["perplexity"] for r in results_by_resource[level].values()]
            avg_perplexity = np.mean(perplexities)
            results_by_resource[level]["_average"] = avg_perplexity
    
    all_perplexities = [r["perplexity"] for r in all_results.values()]
    overall_avg = np.mean(all_perplexities) if all_perplexities else None
    
    return {
        "model_path": model_path,
        "languages": all_results,
        "by_resource": results_by_resource,
        "overall_average": overall_avg,
        "config": {
            "split": split,
            "max_samples_per_lang": max_samples_per_lang,
            "batch_size": batch_size,
            "max_length": max_length
        }
    }


def save_results(results: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(results["model_path"])
    filename = f"culturax_perplexity_results_{model_name}.json"
    json_path = os.path.join(output_dir, filename)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")
    
    csv_rows = []
    for level in ["high", "medium", "low"]:
        for lang, lang_results in results["by_resource"][level].items():
            if lang != "_average":
                csv_rows.append([model_name, level.capitalize(), lang, f"{lang_results['perplexity']:.4f}"])
        if "_average" in results["by_resource"][level]:
            csv_rows.append([model_name, level.capitalize(), "Average", 
                           f"{results['by_resource'][level]['_average']:.4f}"])
    
    if results["overall_average"] is not None:
        csv_rows.append([model_name, "Overall", "Average", f"{results['overall_average']:.4f}"])
    
    csv_path = os.path.join(output_dir, filename.replace(".json", ".csv"))
    df = pd.DataFrame(csv_rows, columns=["Model", "Resource Level", "Language", "Perplexity"])
    df.to_csv(csv_path, index=False)
    print(f"Summary table saved to {csv_path}")
    
    return json_path, csv_path


def print_results_table(results: Dict):
    print("\n" + "="*80)
    print("CulturaX Normalized Perplexity Results")
    print("="*80)
    
    model_name = os.path.basename(results["model_path"])
    print(f"Model: {model_name}")
    print(f"Full Path: {results['model_path']}")
    print()
    
    for level in ["high", "medium", "low"]:
        level_name = level.capitalize() + " Resource"
        print(f"\n{level_name}:")
        print("-" * 80)
        
        for lang in LANGUAGE_RESOURCES[level]:
            if lang in results["by_resource"][level]:
                ppl = results["by_resource"][level][lang]["perplexity"]
                print(f"  {lang:3s}: {ppl:8.4f}")
        
        if "_average" in results["by_resource"][level]:
            avg = results["by_resource"][level]["_average"]
            print(f"  {'Avg':3s}: {avg:8.4f}")
    
    if results["overall_average"] is not None:
        print(f"\nOverall Average: {results['overall_average']:.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate normalized perplexity on CulturaX dataset"
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline Pythia-1B model (EleutherAI/pythia-1b)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="?",
        const="default",
        help="Use experiment model. If no path provided, uses default S2 checkpoint: log/1b/0_qwen2-7b_S2/checkpoint-2500"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Custom model path (can be used alone or overrides --baseline/--experiment)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_LANGUAGES,
        help=f"Languages to evaluate (default: all {len(ALL_LANGUAGES)} languages)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Dataset split to use (default: train - CulturaX only has train split)"
    )
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=None,
        help="Maximum number of samples per language (default: None = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/culturax_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="uonlp/CulturaX",
        help="CulturaX dataset name on HuggingFace"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for perplexity computation (default: 8)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    args = parser.parse_args()
    
    if args.model_path:
        model_path = args.model_path
        print(f"Using custom model path: {model_path}")
    elif args.baseline:
        model_path = "EleutherAI/pythia-1b"
        print(f"Using baseline model: {model_path}")
    elif args.experiment is not None:
        if args.experiment == "default":
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, "log/1b/0_qwen2-7b_S2/checkpoint-2500")
        else:
            model_path = args.experiment
        print(f"Using experiment model: {model_path}")
    else:
        parser.error("Must specify either --baseline, --experiment, or --model_path")
    
    if not model_path.startswith(("http://", "https://")) and not os.path.exists(model_path) and "/" in model_path and not model_path.startswith("EleutherAI/") and not model_path.startswith("Qwen/"):
        print(f"Warning: Model path {model_path} does not exist. It will be loaded from HuggingFace if it's a model ID.")
    
    results = evaluate_all_languages(
        model_path=model_path,
        languages=args.languages,
        split=args.split,
        max_samples_per_lang=args.max_samples_per_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name
    )
    
    print_results_table(results)
    
    save_results(results, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

