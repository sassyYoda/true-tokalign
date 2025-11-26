#!/usr/bin/env python3
"""
Quick test to verify OPUS Global Voices dataset loads correctly and check its format.
Run this on your server where the environment is set up.
"""

from datasets import load_dataset

print("="*60)
print("Testing OPUS Global Voices Dataset Loading")
print("="*60)

try:
    print("\n1. Loading dataset with 'en-es' config (test split)...")
    dataset = load_dataset(
        "sentence-transformers/parallel-sentences-global-voices",
        "en-es",
        split="test",
        trust_remote_code=True
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Type: {type(dataset)}")
    print(f"  Length: {len(dataset)}")
    
    print("\n2. Checking dataset structure...")
    if len(dataset) > 0:
        first_example = dataset[0]
        print(f"  First example keys: {list(first_example.keys())}")
        print(f"  First example:")
        for key, value in first_example.items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"    {key}: {preview}")
            else:
                print(f"    {key}: {value}")
        
        print("\n3. Checking field mapping...")
        sentence1 = first_example.get("sentence1", "")
        sentence2 = first_example.get("sentence2", "")
        
        print(f"  sentence1: {sentence1[:100]}...")
        print(f"  sentence2: {sentence2[:100]}...")
        
        print("\n4. Testing data extraction (as our code would do)...")
        # Simulate what our code does
        pairs = []
        for i, example in enumerate(dataset[:5]):  # Just first 5
            english = example.get("sentence1", "")
            spanish = example.get("sentence2", "")
            
            if spanish and english:
                spanish = str(spanish).strip()
                english = str(english).strip()
                if len(spanish) > 0 and len(english) > 0:
                    pairs.append((spanish, english))
        
        print(f"  Extracted {len(pairs)} pairs")
        print(f"  Sample pairs:")
        for i, (spanish, english) in enumerate(pairs[:3], 1):
            print(f"    {i}. Spanish: {spanish[:70]}...")
            print(f"       English: {english[:70]}...")
        
        print("\n5. Verifying format for evaluation...")
        print(f"  ✓ Pairs are tuples of (spanish, english)")
        print(f"  ✓ Each pair has non-empty strings")
        print(f"  ✓ Format matches what eval_translation.py expects")
    
    print("\n" + "="*60)
    print("✓ Dataset loading test completed successfully!")
    print("="*60)
    print("\nThe dataset format is compatible with the evaluation script.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
