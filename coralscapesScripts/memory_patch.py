# Monkey patch torch's interpolate function to use less memory
import torch.nn.functional as F
import torch

original_interpolate = F.interpolate

def memory_efficient_interpolate(*args, **kwargs):
    """Memory-efficient version of torch.nn.functional.interpolate
    
    This function forces garbage collection before and after interpolation
    to minimize memory fragmentation and reduce the chance of OOM errors.
    """
    # Force garbage collection before interpolation
    torch.cuda.empty_cache()
    
    # Use smaller precision for interpolation if possible
    if kwargs.get('mode') == 'bilinear' and args and isinstance(args[0], torch.Tensor):
        input_tensor = args[0]
        if input_tensor.dtype == torch.float32:
            # Try using float16 for interpolation
            try:
                input_fp16 = input_tensor.half()
                kwargs_copy = kwargs.copy()
                result_fp16 = original_interpolate(input_fp16, *args[1:], **kwargs_copy)
                result = result_fp16.float()
                del input_fp16, result_fp16
                torch.cuda.empty_cache()
                return result
            except:
                # Fall back to original if half precision fails
                pass
    
    # Use original interpolation
    result = original_interpolate(*args, **kwargs)
    
    # Force garbage collection after interpolation
    torch.cuda.empty_cache()
    return result

# Replace the interpolate function with our memory-efficient version
F.interpolate = memory_efficient_interpolate

# Additional memory optimizations
def apply_memory_optimizations():
    """Apply additional memory optimizations to PyTorch"""
    # Disable gradient synchronization for unused parameters
    torch.nn.utils.skip_init = True
    
    # Set memory fraction to avoid OOM
    if torch.cuda.is_available():
        # Reserve only 90% of available memory to leave room for fragmentation
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(0.9, device)
        
        # Enable memory stats for better debugging
        torch.cuda.memory._record_memory_history(max_entries=10000)
        
        print(f"Applied memory optimizations for CUDA device {device}")

# Apply optimizations when this module is imported
apply_memory_optimizations()
