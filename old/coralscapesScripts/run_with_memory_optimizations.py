
# Wrapper script with memory optimizations
import sys

# Load memory patches
with open('memory_patch.py', 'r', encoding='utf-8') as patch_file:
    exec(patch_file.read())

# Load additional memory fixes
with open('extra_memory_fixes.py', 'r', encoding='utf-8') as fixes_file:
    exec(fixes_file.read())

# Set up arguments
sys.argv = ['fine_tune_pipeline.py'] + ['--config', 'configs/segformer-mit-b2_lora']

# Load and run the main pipeline
with open('fine_tune_pipeline.py', 'r', encoding='utf-8') as main_file:
    exec(main_file.read())
        