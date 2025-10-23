#!/usr/bin/env python
"""
Additional memory fixes for the coral bleaching pipeline
"""

import torch
import gc

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to reduce memory usage"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")
    else:
        print("Gradient checkpointing not available for this model")

def reduce_precision(model):
    """Reduce model precision to save memory"""
    try:
        # Try to convert to half precision
        model = model.half()
        print("Converted model to half precision")
        return model
    except Exception as e:
        print(f"Could not convert to half precision: {e}")
        return model

def clear_cache_aggressively():
    """Aggressively clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def patch_torch_operations():
    """Patch PyTorch operations to be more memory efficient"""
    
    # Patch torch.cat to clear cache
    original_cat = torch.cat
    def memory_efficient_cat(*args, **kwargs):
        torch.cuda.empty_cache()
        result = original_cat(*args, **kwargs)
        torch.cuda.empty_cache()
        return result
    torch.cat = memory_efficient_cat
    
    # Patch torch.stack to clear cache
    original_stack = torch.stack
    def memory_efficient_stack(*args, **kwargs):
        torch.cuda.empty_cache()
        result = original_stack(*args, **kwargs)
        torch.cuda.empty_cache()
        return result
    torch.stack = memory_efficient_stack
    
    print("Applied torch operation patches")

def apply_all_memory_fixes(model=None):
    """Apply all memory fixes"""
    print("Applying additional memory fixes...")
    
    # Patch torch operations
    patch_torch_operations()
    
    # Enable gradient checkpointing if model provided
    if model is not None:
        enable_gradient_checkpointing(model)
        # Don't reduce precision as it might cause issues with some models
        # model = reduce_precision(model)
    
    # Clear cache
    clear_cache_aggressively()
    
    print("All memory fixes applied")

if __name__ == "__main__":
    apply_all_memory_fixes()
