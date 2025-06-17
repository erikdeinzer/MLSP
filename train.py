
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
    dataset = cfg.dataset
    optim_cfg = cfg.optim_cfg
    model_cfg = cfg.model_cfg
    

    train_dataloader = cfg.train_dataloader if hasattr(cfg, 'train_dataloader') else None
    val_dataloader = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else None
    test_dataloader = cfg.test_dataloader if hasattr(cfg, 'test_dataloader') else None

    device = cfg.device if hasattr(cfg, 'device') else 'cuda'
    workdir = cfg.work_dir if hasattr(cfg, 'work_dir') else 'results'
    runner_args = cfg.runner_args if hasattr(cfg, 'runner_args') else {}

    runargs = cfg.runargs if hasattr(cfg, 'runargs') else {}
    print("Loaded config:")
    print(f"Batch size: {train_dataloader['batch_size']}")
    print(f"Optimizer type: {optim_cfg['type']}")
    print(f"Model head output dims: {model_cfg['head_cfg']['odims']}")

    runner = Runner(
        model=model_cfg, 
        dataset=dataset,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optim=optim_cfg, 
        device=device, 
        work_dir=workdir, **runner_args)
    
    runner.run(mode='train', val_interval=1, log_interval=1, epochs=100, start_epoch=1, **runargs)

