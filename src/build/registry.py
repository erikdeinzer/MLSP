from typing import Callable, List, Union
import torchvision.transforms as T
import inspect
import torch.nn as nn
import torch


class Registry():
    """
    A simple registry to register and retrieve objects by name.
    This is useful for managing different components and simplifies the process of
    instantiating them based on configuration files.
    The registry can be used to register models, optimizers, and other components
    """
    def __init__(self, name='default'):
        self.name = name
        self._registry = {}

    def register_module(self, obj=None, *, type=None):
        if obj is None:
            return lambda obj: self.register_module(obj, type=type)
        key = type or obj.__name__
        if key in self._registry:
            raise ValueError('Duplicate registrations with same name not possible')
        self._registry[key]=obj
        return obj
    
    def get(self, type):
        if type not in self._registry:
            raise KeyError(f"'{type}' not found in the '{self.name}' registry.")
        obj = self._registry.get(type)
        return obj
    
    def all(self):
        return self._registry
    
    def __contains__(self, type):
        return type in self._registry

    def __len__(self):
        return len(self._registry)
    

MODULES = Registry('MODELS')
EVALUATORS = Registry('EVALUATIONS')
TRANSFORMS = Registry('TRANSFORMS')
OPTIMIZERS = Registry('OPTIMIZERS')
DATASETS = Registry('DATASETS')



def build_module_from_registry(cfg: dict, registry, **kwargs) -> object:
    """
    Build a module from a configuration dictionary.
    Args:
        cfg: Dictionary containing the configuration for the module.
        registry: The registry to use for building the module.
        **kwargs: Additional keyword arguments to pass to the module constructor.
    Returns:
        An instance of the module.
    """
    cfg = cfg.copy()

    cfg.update(kwargs)
    cls = registry.get(cfg.pop('type'))
    return cls(**cfg)

def build_module(module, registry=None,**kwargs):
    if isinstance(module, nn.Module):
        return module
    elif isinstance(module, dict):
        cls = module.get('type')
        if isinstance(cls, type):
            params = {k: v for k, v in module.items() if k != 'type'}
            return cls(**params, **kwargs)
        if isinstance(cls, str):
            if registry is None:
                raise ValueError("Registry must be provided for string type modules.")
            mod = build_module_from_registry(module, registry, **kwargs)
            if mod is None:
                raise ValueError(f"Module '{cls}' not found in the registry.")
            return mod
    else:
        raise TypeError(f"Unsupported module type: {type(module)}. Expected nn.Module or dict.")







def register_from_module(module, registry, base_class=None, verbose=False, filter_fn=None):
    """
    Register classes from a module to a registry.
    Args:
        module: The module to register classes from.
        registry: The registry to register the classes to.
        base_class: The base class that the registered classes should inherit from.
        verbose: Whether to print the registration process.
        filter_fn: A function to filter the classes to be registered.
    """

    for name in dir(module):  # Use dir to get full visible symbol list
        try:
            obj = getattr(module, name)  # This triggers lazy imports
        except Exception:
            continue
        if not inspect.isclass(obj):
            continue
        try:
            if base_class is not None and not issubclass(obj, base_class):
                continue
        except TypeError:
            continue
        if filter_fn and not filter_fn(obj):
            continue
        try:
            inspect.signature(obj)  # Just test it's introspectable
            registry.register_module(obj)
            if verbose:
                print(f"[{registry.name}] Registered: {name}")
        except Exception:
            continue
# Register PyTorch models
register_from_module(nn, MODULES, base_class=torch.nn.Module, verbose=False)
# Register PyTorch transforms
register_from_module(T, TRANSFORMS, verbose=False)
# Register PyTorch optimizers
register_from_module(torch.optim, OPTIMIZERS, base_class=torch.optim.Optimizer, verbose=False)




