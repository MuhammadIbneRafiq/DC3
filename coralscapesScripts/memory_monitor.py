#!/usr/bin/env python
"""
Memory monitoring utility for debugging CUDA memory issues
"""
import os
import torch
import psutil
import gc

# Set CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def print_memory_info():
    """Print detailed memory information"""
    print("=" * 60)
    print("MEMORY STATUS")
    print("=" * 60)
    
    # System memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"System RAM: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    # GPU memory
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")
            
            # Memory fragmentation info
            memory_stats = torch.cuda.memory_stats(i)
            print(f"  Active Memory: {memory_stats['active_bytes.all.current'] / 1024**3:.2f} GB")
            print(f"  Inactive Memory: {memory_stats['inactive_split_bytes.all.current'] / 1024**3:.2f} GB")
    else:
        print("CUDA not available")
    
    print("=" * 60)

def cleanup_and_monitor():
    """Clean up memory and monitor the process"""
    print("Before cleanup:")
    print_memory_info()
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    print("\nAfter cleanup:")
    print_memory_info()

def test_model_loading():
    """Test loading a small model to check memory usage"""
    print("Testing model loading...")
    
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        # Try loading a smaller model first
        print("Loading DINOv2-base (smaller model)...")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        print("Model loaded successfully!")
        print_memory_info()
        
        # Clean up
        del model, processor
        cleanup_and_monitor()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print_memory_info()

if __name__ == "__main__":
    print("CUDA Memory Monitor")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    cleanup_and_monitor()
    test_model_loading()
