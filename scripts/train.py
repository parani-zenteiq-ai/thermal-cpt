#!/usr/bin/env python
import torch
import sys
sys.path.insert(0, '/home/parani/thermal-cpt')

from src.utils.config import load_config
from src.utils.distributed import setup_distributed, cleanup_distributed, print_rank_0
from src.model.load import load_model, wrap_model_fsdp
from src.data.dataloader import create_dataloader
from src.training.trainer import Trainer
from src.utils.logging import Logger
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

def main():
    # Load config
    config = load_config("configs/training/phase1_warmup.yaml")
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    print_rank_0(f"Starting training on {world_size} GPUs")
    
    # Load model
    model_cfg = load_config(config.model.config)
    model = load_model(model_cfg)
    
    # Wrap with FSDP
    fsdp_cfg = load_config(config.distributed.config)
    model = wrap_model_fsdp(model, fsdp_cfg)
    
    # Create dataloader
    dataloader = create_dataloader(
        "data/processed/fineweb_edu/tokenized",
        batch_size=config.training.micro_batch_size
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        betas=config.training.optimizer.betas,
        weight_decay=config.training.optimizer.weight_decay
    )
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.scheduler.warmup_steps,
        num_training_steps=config.training.max_steps
    )
    
    # Logger
    logger = Logger(
        project=config.logging.wandb_project,
        run_name=config.logging.wandb_run_name,
        config=config.to_dict()
    )
    
    # Trainer
    trainer = Trainer(config, model, dataloader, optimizer, scheduler, logger)
    
    # Train
    print_rank_0("Starting training...")
    trainer.train()
    
    cleanup_distributed()
    logger.finish()
    print_rank_0("✓ Training complete!")

if __name__ == "__main__":
    main()
