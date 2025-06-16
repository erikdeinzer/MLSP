
import importlib.util
import sys
import os
from types import ModuleType
from src.runner import Runner


def load_config_module(path: str) -> ModuleType:
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    cfg_module = importlib.util.module_from_spec(spec)
    sys.modules["cfg_module"] = cfg_module
    spec.loader.exec_module(cfg_module)
    return cfg_module


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py path/to/config.py")
    
    config_path = sys.argv[1]
    cfg = load_config_module(config_path)

    # Access your configs:
    loading_cfg = cfg.loading_cfg
    optim_cfg = cfg.optim_cfg
    model_cfg = cfg.model_cfg
    dataset_cfg = cfg.dataset_cfg
    device = cfg.device if hasattr(cfg, 'device') else 'cuda'
    workdir = cfg.work_dir if hasattr(cfg, 'work_dir') else 'results'

    print("Loaded config:")
    print(f"Batch size: {loading_cfg['batch_size']}")
    print(f"Optimizer type: {optim_cfg['type']}")
    print(f"Model head output dims: {model_cfg['head_cfg']['odims']}")

    runner = Runner(
        model=model_cfg, 
        dataloader_cfg=loading_cfg, 
        dataset=dataset_cfg, 
        optim=optim_cfg, 
        device=device, 
        work_dir=workdir)
    
    runner.run(mode='train', val_interval=1, log_interval=1, epochs=100, start_epoch=1)

