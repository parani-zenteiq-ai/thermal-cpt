import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenize_and_pack(input_file: str, output_dir: str, seq_length: int = 4096):
    """Tokenize text and pack into fixed-length sequences"""
    print(f"Tokenizing {input_file}...")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_tokens = []
    total_docs = 0
    
    # Read and tokenize in chunks
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc="Tokenizing"):
            text = line.strip()
            if not text:
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)  # Document separator
            all_tokens.extend(tokens)
            total_docs += 1
            
            # Save in chunks to avoid memory issues
            if len(all_tokens) >= 100_000_000:  # 100M tokens
                _save_chunk(all_tokens, output_path, seq_length)
                all_tokens = []
    
    # Save remaining
    if all_tokens:
        _save_chunk(all_tokens, output_path, seq_length)
    
    print(f"✓ Tokenized {total_docs} documents")

def _save_chunk(tokens, output_path, seq_length):
    """Save tokens as packed sequences"""
    # Pack into sequences
    num_sequences = len(tokens) // seq_length
    tokens = tokens[:num_sequences * seq_length]  # Drop incomplete
    sequences = np.array(tokens, dtype=np.int32).reshape(-1, seq_length)
    
    # Save with unique name
    chunk_id = len(list(output_path.glob("*.npy")))
    np.save(output_path / f"chunk_{chunk_id:04d}.npy", sequences)
    print(f"  Saved chunk {chunk_id}: {len(sequences)} sequences")

if __name__ == "__main__":
    tokenize_and_pack(
        "data/processed/fineweb_edu/fineweb_edu_clean.txt",
        "data/processed/fineweb_edu/tokenized",
        seq_length=4096
    )
