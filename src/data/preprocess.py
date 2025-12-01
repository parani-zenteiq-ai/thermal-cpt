import os
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

class MemoryEfficientPreprocessor:
    """Stream-based preprocessing - never loads full dataset in RAM"""
    
    def __init__(self, config):
        self.config = config
        self.lsh = MinHashLSH(threshold=config.deduplication.threshold, 
                              num_perm=config.deduplication.num_perm)
    
    def deduplicate_streaming(self, input_file: str, output_file: str):
        """Deduplicate without loading full file in memory"""
        print(f"Deduplicating {input_file}...")
        
        seen_hashes = set()
        unique_count = 0
        total_count = 0
        
        # Process in chunks
        CHUNK_SIZE = 10000  # Process 10k docs at a time
        
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            chunk = []
            
            for line in tqdm(fin):
                chunk.append(line.strip())
                
                if len(chunk) >= CHUNK_SIZE:
                    unique_count += self._process_chunk(chunk, fout, seen_hashes)
                    total_count += len(chunk)
                    chunk = []  # Clear chunk from memory
            
            # Process remaining
            if chunk:
                unique_count += self._process_chunk(chunk, fout, seen_hashes)
                total_count += len(chunk)
        
        dedup_rate = (1 - unique_count/total_count) * 100
        print(f"✓ Kept {unique_count}/{total_count} ({dedup_rate:.1f}% removed)")
    
    def _process_chunk(self, chunk, fout, seen_hashes):
        """Process chunk and return unique count"""
        unique_in_chunk = 0
        
        for doc in chunk:
            # Create hash
            m = MinHash(num_perm=128)
            for word in doc.split():
                m.update(word.encode('utf8'))
            
            hash_val = tuple(m.hashvalues)
            
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                fout.write(doc + '\n')
                unique_in_chunk += 1
        
        return unique_in_chunk
    
    def filter_quality(self, input_file: str, output_file: str):
        """Filter low-quality text - streaming"""
        print(f"Quality filtering {input_file}...")
        
        kept = 0
        total = 0
        
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in tqdm(fin):
                text = line.strip()
                total += 1
                
                # Quick quality checks
                words = text.split()
                if len(words) < 50:  # Too short
                    continue
                
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < self.config.quality_filters.min_unique_token_ratio:
                    continue
                
                special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
                if special_chars / len(text) > self.config.quality_filters.max_special_char_ratio:
                    continue
                
                fout.write(text + '\n')
                kept += 1
        
        filter_rate = (1 - kept/total) * 100
        print(f"✓ Kept {kept}/{total} ({filter_rate:.1f}% filtered)")

def preprocess_corpus(source_name: str):
    """Preprocess a single data source"""
    from src.utils.config import load_config
    
    preproc_cfg = load_config("configs/data/preprocessing.yaml")
    sources_cfg = load_config("configs/data/sources.yaml")
    
    # Get paths
    source_config = getattr(sources_cfg, source_name)
    raw_dir = Path(source_config.output_dir)
    processed_dir = Path("data/processed") / source_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input file
    input_files = list(raw_dir.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No .txt files in {raw_dir}")
    
    preprocessor = MemoryEfficientPreprocessor(preproc_cfg)
    
    for input_file in input_files:
        print(f"\nProcessing {input_file.name}...")
        
        # Step 1: Deduplicate
        dedup_file = processed_dir / f"{input_file.stem}_dedup.txt"
        preprocessor.deduplicate_streaming(str(input_file), str(dedup_file))
        
        # Step 2: Quality filter
        final_file = processed_dir / f"{input_file.stem}_clean.txt"
        preprocessor.filter_quality(str(dedup_file), str(final_file))
        
        # Remove intermediate file to save space
        dedup_file.unlink()
        
        print(f"✓ Final output: {final_file}")

if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "fineweb_edu"
    preprocess_corpus(source)
