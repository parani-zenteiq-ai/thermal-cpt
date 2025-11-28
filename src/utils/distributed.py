import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is rank 0"""
    return not dist.is_initialized() or dist.get_rank() == 0

def print_rank_0(message):
    """Print only from main process"""
    if is_main_process():
        print(message)
