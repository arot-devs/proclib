#!/bin/bash
pip install -U pip --cache-dir /rmt/nyanko/pip-cache
pip install xformers "torch<2.4" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -U 
pip install -U lightning transformers fairscale wandb tqdm pip diffusers huggingface_hub hf_transfer 
pip install open_clip_torch==2.24.0 --cache-dir /rmt/nyanko/pip-cache