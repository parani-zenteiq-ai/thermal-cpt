import torch
from transformers import AutoModelForCausalLM, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def load_model(model_config):
    """Load pretrained model"""
    print(f"Loading model: {model_config.name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    return model

def wrap_model_fsdp(model, fsdp_config):
    """Wrap model with FSDP"""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=False,
    )
    
    print("✓ Model wrapped with FSDP")
    return model
