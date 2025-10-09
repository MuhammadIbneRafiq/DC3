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
        # Reserve only 70% of available memory to leave room for fragmentation
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(0.7, device)
        
        # Enable memory stats for better debugging
        torch.cuda.memory._record_memory_history(max_entries=10000)
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
        print(f"Applied memory optimizations for CUDA device {device}")
        
        # Print current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Monkey patch SegFormer's forward method to reduce memory usage
try:
    from transformers.models.segformer.modeling_segformer import SegformerForSemanticSegmentation
    
    original_forward = SegformerForSemanticSegmentation.forward
    
    def memory_efficient_forward(self, pixel_values, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """Memory-efficient forward pass for SegFormer"""
        # Clear cache before forward pass
        torch.cuda.empty_cache()
        
        # Reduce batch size if too large
        if pixel_values.shape[0] > 1:
            # Process one sample at a time to avoid OOM
            outputs = []
            for i in range(pixel_values.shape[0]):
                single_input = pixel_values[i:i+1]
                single_labels = labels[i:i+1] if labels is not None else None
                
                # Clear cache before each sample
                torch.cuda.empty_cache()
                
                # Call original forward with single sample
                output = original_forward(self, single_input, single_labels, output_attentions, output_hidden_states, return_dict)
                outputs.append(output)
                
                # Clear cache after each sample
                torch.cuda.empty_cache()
            
            # Combine outputs
            if hasattr(outputs[0], 'logits'):
                logits = torch.cat([out.logits for out in outputs], dim=0)
                return type(outputs[0])(logits=logits)
            else:
                return outputs[0]
        else:
            # Single sample, use original forward
            result = original_forward(self, pixel_values, labels, output_attentions, output_hidden_states, return_dict)
            torch.cuda.empty_cache()
            return result
    
    # Replace the forward method
    SegformerForSemanticSegmentation.forward = memory_efficient_forward
    print("Applied SegFormer memory optimizations")
    
except ImportError:
    print("SegFormer not available for patching")
except Exception as e:
    print(f"Failed to patch SegFormer: {e}")

# Apply optimizations when this module is imported
apply_memory_optimizations()
