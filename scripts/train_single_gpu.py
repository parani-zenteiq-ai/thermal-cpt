#!/usr/bin/env python
import torch
import sys
sys.path.insert(0, '/home/parani/thermal-cpt')

from src.utils.config import load_config
from src.model.load import load_model
from src.data.dataloader import create_dataloader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def main():
    config = load_config("configs/training/phase1_warmup.yaml")
    
    print("Loading model...")
    model_cfg = load_config(config.model.config)
    model = load_model(model_cfg)
    model = model.cuda()
    
    print("Creating dataloader...")
    dataloader = create_dataloader(
        "data/processed/fineweb_edu/tokenized",
        batch_size=1  # Reduced for single GPU
    )
    
    print("Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        betas=config.training.optimizer.betas,
        weight_decay=config.training.optimizer.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.scheduler.warmup_steps,
        num_training_steps=config.training.max_steps
    )
    
    print("Starting training...")
    model.train()
    
    for step, batch in enumerate(tqdm(dataloader, total=100)):
        input_ids = batch["input_ids"].cuda()
        
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        if step >= 100:
            break
    
    print("✓ Test training complete!")

if __name__ == "__main__":
    main()
