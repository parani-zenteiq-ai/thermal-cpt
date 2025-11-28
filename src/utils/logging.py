import os
import wandb
from typing import Dict, Any, Optional
from .distributed import is_main_process

class Logger:
    """Unified logger for tensorboard and wandb"""
    
    def __init__(self, project: str, run_name: str, config: Dict[str, Any]):
        self.enabled = is_main_process()
        
        if self.enabled:
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                resume="allow"
            )
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics"""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def finish(self):
        """Close logger"""
        if self.enabled:
            wandb.finish()
