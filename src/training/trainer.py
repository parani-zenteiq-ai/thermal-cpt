import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from pathlib import Path

class Trainer:
    def __init__(self, config, model, train_dataloader, optimizer, scheduler, logger):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        
        self.global_step = 0
        self.tokens_seen = 0
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        for batch in self.train_dataloader:
            loss = self.training_step(batch)
            
            if self.global_step % self.config.logging.log_every_steps == 0:
                self.log_metrics(loss)
            
            if self.should_save_checkpoint():
                self.save_checkpoint()
            
            self.global_step += 1
            
            if self.global_step >= self.config.training.max_steps:
                break
    
    def training_step(self, batch):
        """Single training step"""
        input_ids = batch["input_ids"].cuda()
        labels = input_ids.clone()
        
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.tokens_seen += input_ids.numel()
        
        return loss.item()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        if dist.get_rank() == 0:
            save_dir = Path(self.config.checkpointing.save_dir) / f"step_{self.global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FSDP model
            self.model.save_pretrained(save_dir)
            print(f"✓ Checkpoint saved: {save_dir}")
    
    def should_save_checkpoint(self):
        return self.global_step % self.config.checkpointing.save_every_steps == 0
    
    def log_metrics(self, loss):
        metrics = {
            "loss": loss,
            "lr": self.scheduler.get_last_lr()[0],
            "tokens": self.tokens_seen,
            "step": self.global_step
        }
        self.logger.log(metrics, self.global_step)
        print(f"Step {self.global_step} | Loss: {loss:.4f} | Tokens: {self.tokens_seen:,}")
