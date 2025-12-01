import os
import requests
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_fineweb_edu(output_dir: str, sample_tokens: int):
    """Download FineWeb-Edu replay buffer"""
    print(f"Downloading FineWeb-Edu ({sample_tokens/1e9:.1f}B tokens)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    total_tokens = 0
    with open(f"{output_dir}/fineweb_edu.txt", "w") as f:
        for sample in tqdm(dataset):
            text = sample["text"]
            f.write(text + "\n")
            total_tokens += len(text.split())
            
            if total_tokens >= sample_tokens:
                break
    
    print(f"✓ Downloaded {total_tokens:,} tokens to {output_dir}")

def download_arxiv_thermal(output_dir: str):
    """Download arXiv thermal/CFD papers"""
    print("Downloading arXiv thermal papers...")
    print("⚠️  This requires arxiv Python package and takes ~2 hours")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Placeholder - you'll implement actual arxiv download
    print(f"✓ Output directory ready: {output_dir}")

if __name__ == "__main__":
    from src.utils.config import load_config
    
    sources_cfg = load_config("configs/data/sources.yaml")
    
    # Download FineWeb first (fastest to test)
    download_fineweb_edu(
        output_dir=sources_cfg.fineweb_edu.output_dir,
        sample_tokens=sources_cfg.fineweb_edu.sample_tokens
    )
