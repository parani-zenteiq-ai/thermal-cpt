import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized sequences"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.chunk_files = sorted(list(self.data_dir.glob("chunk_*.npy")))
        
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
        
        # Load all chunks into memory (2GB is manageable)
        self.sequences = []
        for chunk_file in self.chunk_files:
            chunk = np.load(chunk_file)
            self.sequences.append(chunk)
        
        self.sequences = np.concatenate(self.sequences, axis=0)
        print(f"✓ Loaded {len(self.sequences)} sequences from {len(self.chunk_files)} chunks")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long)
        }

def create_dataloader(data_dir, batch_size, num_workers=4):
    """Create dataloader for training"""
    dataset = TokenizedDataset(data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader
