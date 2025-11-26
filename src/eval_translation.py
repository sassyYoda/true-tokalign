"""
Translation evaluation script for Spanish-English translation using OPUS Global Voices corpus.
Evaluates the trained model on translation tasks in both directions and computes BLEU scores.
"""

import argparse
import json
import os
import zipfile
import urllib.request
from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sacrebleu
from sacrebleu.metrics import BLEU

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: Optional[torch.dtype] = None
):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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


def create_translation_prompt(
    source_text: str,
    direction: str,
    prompt_template: Optional[str] = None
) -> str:
    """Create a prompt for translation task."""
    if prompt_template:
        # Use custom template with {source} placeholder
        prompt = prompt_template.replace("{source}", source_text)
    elif direction == "es-en":
        prompt = f"Translate the following Spanish text to English:\n\nSpanish: {source_text}\nEnglish:"
    elif direction == "en-es":
        prompt = f"Translate the following English text to Spanish:\n\nEnglish: {source_text}\nSpanish:"
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return prompt


def generate_translation(
    model,
    tokenizer,
    source_text: str,
    direction: str = "es-en",
    prompt_template: Optional[str] = None,
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """Generate translation using the model with a prompt."""
    # Create prompt
    prompt = create_translation_prompt(source_text, direction, prompt_template)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (remove input)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return translation.strip()


def download_opus_globalvoices(
    data_dir: str = "./data/GlobalVoices",
    force_download: bool = True
) -> Tuple[str, str]:
    """Download OPUS Global Voices Spanish-English parallel corpus."""
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for OPUS Global Voices Spanish-English
    zip_url = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/es-en.txt.zip"
    zip_path = os.path.join(data_dir, "es-en.txt.zip")
    es_file = os.path.join(data_dir, "GlobalVoices.es-en.es")
    en_file = os.path.join(data_dir, "GlobalVoices.es-en.en")
    
    # Check if files already exist (local files take precedence)
    if os.path.exists(es_file) and os.path.exists(en_file):
        if not force_download:
            print(f"✓ OPUS Global Voices files found at {data_dir}")
            print(f"  Spanish: {es_file}")
            print(f"  English: {en_file}")
            return es_file, en_file
        else:
            print(f"Force download enabled (default), will re-download dataset...")
    
    # Download zip file
    if not os.path.exists(zip_path) or force_download:
        print(f"Downloading OPUS Global Voices from {zip_url}...")
        print(f"Saving to {zip_path}...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) / total_size) if total_size > 0 else 0
            print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)
        
        try:
            urllib.request.urlretrieve(zip_url, zip_path, show_progress)
            print()  # New line after progress
        except Exception as e:
            raise Exception(f"Failed to download OPUS Global Voices: {e}")
    
    # Extract zip file
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted to {data_dir}")
    except Exception as e:
        raise Exception(f"Failed to extract zip file: {e}")
    
    # Verify files exist
    if not os.path.exists(es_file) or not os.path.exists(en_file):
        raise Exception(f"Extracted files not found. Expected {es_file} and {en_file}")
    
    print(f"✓ Spanish file: {es_file}")
    print(f"✓ English file: {en_file}")
    
    return es_file, en_file


def load_opus_globalvoices(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    force_download: bool = True
) -> List[Tuple[str, str]]:
    """Load OPUS Global Voices Spanish-English parallel corpus from local files."""
    print(f"Loading OPUS Global Voices dataset (split: {split})...")
    
    # Set data directory
    if data_dir is None:
        data_dir = cache_dir if cache_dir else "./data/GlobalVoices"
    
    # Download if needed
    es_file, en_file = download_opus_globalvoices(data_dir, force_download)
    
    # Read parallel files
    print(f"Reading Spanish file: {es_file}")
    print(f"Reading English file: {en_file}")
    
    pairs = []
    with open(es_file, "r", encoding="utf-8") as f_es, \
         open(en_file, "r", encoding="utf-8") as f_en:
        
        spanish_lines = f_es.readlines()
        english_lines = f_en.readlines()
    
    # Verify same number of lines
    if len(spanish_lines) != len(english_lines):
        print(f"Warning: Mismatched line counts - Spanish: {len(spanish_lines)}, English: {len(english_lines)}")
        min_lines = min(len(spanish_lines), len(english_lines))
        spanish_lines = spanish_lines[:min_lines]
        english_lines = english_lines[:min_lines]
    
    # Create pairs
    total_lines = len(spanish_lines)
    print(f"Found {total_lines:,} parallel sentence pairs")
    
    # Handle train/test/validation splits
    if split == "test":
        # Use last 10% for test
        start_idx = int(total_lines * 0.9)
        end_idx = total_lines
    elif split == "validation":
        # Use 10-20% for validation
        start_idx = int(total_lines * 0.1)
        end_idx = int(total_lines * 0.2)
    elif split == "train":
        # Use first 90% for training
        start_idx = 0
        end_idx = int(total_lines * 0.9)
    else:
        # Use all data
        start_idx = 0
        end_idx = total_lines
    
    print(f"Using split '{split}': lines {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} pairs)")
    
    # Extract pairs
    for i in range(start_idx, end_idx):
        spanish = spanish_lines[i].strip()
        english = english_lines[i].strip()
        
        # Skip empty lines
        if len(spanish) > 0 and len(english) > 0:
            pairs.append((spanish, english))
        
        if max_samples and len(pairs) >= max_samples:
            break
    
    print(f"Loaded {len(pairs)} sentence pairs")
    return pairs


def evaluate_translation(
    model,
    tokenizer,
    source_texts: List[str],
    target_texts: List[str],
    direction: str,
    prompt_template: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> dict:
    """Evaluate translation quality using BLEU scores."""
    print(f"\nEvaluating {direction} translation...")
    
    if max_samples:
        source_texts = source_texts[:max_samples]
        target_texts = target_texts[:max_samples]
    
    predictions = []
    references = []
    
    # Generate translations
    for i in tqdm(range(len(source_texts)), desc=f"Generating {direction} translations"):
        source = source_texts[i]
        reference = target_texts[i]
        
        try:
            prediction = generate_translation(
                model, tokenizer, source, direction=direction,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device
            )
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(f"Error translating example {i}: {e}")
            predictions.append("")
            references.append(reference)
    
    # Compute BLEU scores
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    
    # Also compute sentence-level BLEU for statistics
    sentence_bleus = []
    for pred, ref in zip(predictions, references):
        try:
            sent_bleu = bleu.sentence_score(pred, [ref])
            sentence_bleus.append(sent_bleu.score)
        except:
            sentence_bleus.append(0.0)
    
    avg_sentence_bleu = sum(sentence_bleus) / len(sentence_bleus) if sentence_bleus else 0.0
    
    results = {
        "direction": direction,
        "num_examples": len(predictions),
        "corpus_bleu": bleu_score.score,
        "corpus_bleu_details": {
            "precisions": bleu_score.precisions,
            "bp": bleu_score.bp,
            "ratio": bleu_score.ratio,
            "hyp_len": bleu_score.hyp_len,
            "ref_len": bleu_score.ref_len,
        },
        "avg_sentence_bleu": avg_sentence_bleu,
    }
    
    return results, predictions, references


def save_results(
    results: dict,
    predictions: List[str],
    references: List[str],
    output_dir: str,
    direction: str
):
    """Save evaluation results and translations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{direction}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save translations (predictions and references)
    translations_path = os.path.join(output_dir, f"{direction}_translations.txt")
    with open(translations_path, "w", encoding="utf-8") as f:
        for pred, ref in zip(predictions, references):
            f.write(f"PRED: {pred}\n")
            f.write(f"REF:  {ref}\n")
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translation quality on OPUS Global Voices corpus"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use (train, validation, test)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/translation_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets (also used as data directory for OPUS)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory to store/download OPUS Global Voices dataset (default: ./data/GlobalVoices)"
    )
    parser.add_argument(
        "--no_force_download",
        action="store_true",
        help="Don't force re-download (use existing files if they exist)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="both",
        choices=["both", "es-en", "en-es"],
        help="Translation directions to evaluate"
    )
    parser.add_argument(
        "--prompt_template_es_en",
        type=str,
        default=None,
        help="Custom prompt template for Spanish->English (use {source} placeholder)"
    )
    parser.add_argument(
        "--prompt_template_en_es",
        type=str,
        default=None,
        help="Custom prompt template for English->Spanish (use {source} placeholder)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    force_download = not args.no_force_download  # Default to True unless --no_force_download is set
    pairs = load_opus_globalvoices(
        split=args.dataset_split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        force_download=force_download
    )
    
    if not pairs:
        raise ValueError("No sentence pairs loaded from dataset")
    
    spanish_texts = [pair[0] for pair in pairs]
    english_texts = [pair[1] for pair in pairs]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    all_results = {}
    
    # Evaluate Spanish -> English
    if args.directions in ["both", "es-en"]:
        es_en_results, es_en_preds, es_en_refs = evaluate_translation(
            model, tokenizer,
            spanish_texts, english_texts,
            direction="es-en",
            prompt_template=args.prompt_template_es_en,
            max_samples=args.max_samples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        all_results["es-en"] = es_en_results
        save_results(
            es_en_results, es_en_preds, es_en_refs,
            args.output_dir, "es-en"
        )
        
        print(f"\n{'='*60}")
        print(f"Spanish -> English Results:")
        print(f"{'='*60}")
        print(f"Corpus BLEU: {es_en_results['corpus_bleu']:.4f}")
        print(f"BLEU-1: {es_en_results['corpus_bleu_details']['precisions'][0]:.4f}")
        print(f"BLEU-2: {es_en_results['corpus_bleu_details']['precisions'][1]:.4f}")
        print(f"BLEU-3: {es_en_results['corpus_bleu_details']['precisions'][2]:.4f}")
        print(f"BLEU-4: {es_en_results['corpus_bleu_details']['precisions'][3]:.4f}")
        print(f"Average Sentence BLEU: {es_en_results['avg_sentence_bleu']:.4f}")
        print(f"{'='*60}\n")
    
    # Evaluate English -> Spanish
    if args.directions in ["both", "en-es"]:
        en_es_results, en_es_preds, en_es_refs = evaluate_translation(
            model, tokenizer,
            english_texts, spanish_texts,
            direction="en-es",
            prompt_template=args.prompt_template_en_es,
            max_samples=args.max_samples,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        all_results["en-es"] = en_es_results
        save_results(
            en_es_results, en_es_preds, en_es_refs,
            args.output_dir, "en-es"
        )
        
        print(f"\n{'='*60}")
        print(f"English -> Spanish Results:")
        print(f"{'='*60}")
        print(f"Corpus BLEU: {en_es_results['corpus_bleu']:.4f}")
        print(f"BLEU-1: {en_es_results['corpus_bleu_details']['precisions'][0]:.4f}")
        print(f"BLEU-2: {en_es_results['corpus_bleu_details']['precisions'][1]:.4f}")
        print(f"BLEU-3: {en_es_results['corpus_bleu_details']['precisions'][2]:.4f}")
        print(f"BLEU-4: {en_es_results['corpus_bleu_details']['precisions'][3]:.4f}")
        print(f"Average Sentence BLEU: {en_es_results['avg_sentence_bleu']:.4f}")
        print(f"{'='*60}\n")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

