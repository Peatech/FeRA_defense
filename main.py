"""
Main entry point.
"""
import os
import hydra
import omegaconf
import traceback

# Set CUDA_VISIBLE_DEVICES early, before PyTorch import
# This must happen before any PyTorch operations
def setup_cuda_devices():
    """Set CUDA_VISIBLE_DEVICES before PyTorch initialization."""
    # Check if SLURM allocated GPUs
    slurm_gpus = os.environ.get('SLURM_JOB_GPUS')
    if slurm_gpus:
        # SLURM allocated specific GPUs, but we need to map them to available GPUs
        # Check what GPUs are actually available via nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_gpus = [idx.strip() for idx in result.stdout.strip().split('\n') if idx.strip()]
                if available_gpus:
                    # Use the first available GPU
                    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus[0]
                    print(f"SLURM allocated {slurm_gpus} GPUs, using available GPU: {available_gpus[0]}")
                    return
        except Exception as e:
            print(f"Could not query available GPUs: {e}")
    
    # Check if CUDA_VISIBLE_DEVICES is already set (e.g., by SLURM)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        # Validate that the GPU actually exists
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_gpus = [idx.strip() for idx in result.stdout.strip().split('\n') if idx.strip()]
                current_gpu = os.environ['CUDA_VISIBLE_DEVICES']
                if current_gpu not in available_gpus:
                    # Use the first available GPU instead
                    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus[0]
                    print(f"GPU {current_gpu} not available, using GPU: {available_gpus[0]}")
        except Exception:
            pass
        return
    
    # Read config to get default GPU setting
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load('config/cifar10.yaml')
        if hasattr(config, 'cuda_visible_devices') and config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    except Exception:
        # If config loading fails, continue without setting CUDA_VISIBLE_DEVICES
        pass

# Setup CUDA devices before any PyTorch imports
setup_cuda_devices()

import torch
import ray

from hydra.core.hydra_config import HydraConfig
from rich.traceback import install
from backfed.servers.base_server import BaseServer
from backfed.utils import system_startup, log
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from logging import ERROR

# Use a decorator that allows specifying the config file via command line
# To change main config file: 
#   python main.py --config-name (or -cn) sentiment140 (corresponds to config/sentiment140.yaml) 
@hydra.main(config_path="config", config_name="cifar10", version_base=None)
def main(config: DictConfig):
    # Add output_dir to config
    hydra_cfg = HydraConfig.get()
    with open_dict(config):
        config.output_dir = hydra_cfg.runtime.output_dir
    # Set seeds and do some setups
    system_startup(config)
    aggregator = config["aggregator"]
    try:
        server : BaseServer = instantiate(config.aggregator_config[aggregator], server_config=config, _recursive_=False)
        server.run_experiment()
    except Exception as e:
        error_traceback = traceback.format_exc()
        log(ERROR, f"Error: {e}\n{error_traceback}") # Log traceback
        exit(1)

if __name__ == "__main__":
    # Rich traceback and suppress traceback from hydra, omegaconf, and torch
    OmegaConf.register_new_resolver("eval", eval) # For conditional config on dir_tag
    install(show_locals=False, suppress=[hydra, omegaconf, torch, ray])
    os.environ["HYDRA_FULL_ERROR"] = "1" # For detailed error messages
    os.environ["RAY_memory_monitor_refresh_ms"] = '0' # Disable worker killing
    main()
