import time
import torch
from typing import Optional

class TrainingMonitor:
    """Monitor training progress and throughput"""
    
    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens
        self.tokens_processed = 0
        self.start_time = time.time()
        self.step_start_time = time.time()
    
    def update(self, tokens_in_batch: int):
        """Update token count"""
        self.tokens_processed += tokens_in_batch
    
    def get_stats(self) -> dict:
        """Get current training statistics"""
        elapsed = time.time() - self.start_time
        step_time = time.time() - self.step_start_time
        self.step_start_time = time.time()
        
        tokens_per_sec = self.tokens_processed / elapsed if elapsed > 0 else 0
        progress = self.tokens_processed / self.total_tokens
        
        remaining_tokens = self.total_tokens - self.tokens_processed
        eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0
        
        return {
            "tokens_processed": self.tokens_processed,
            "tokens_per_sec": tokens_per_sec,
            "progress": progress,
            "eta_hours": eta_seconds / 3600,
            "step_time": step_time,
            "gpu0_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            "gpu1_memory_gb": torch.cuda.memory_allocated(1) / 1e9 if torch.cuda.device_count() > 1 else 0
        }
