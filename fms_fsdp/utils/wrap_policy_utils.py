import copy
from torch.nn import Module
from typing import Any, Callable, Union

from torch.distributed.fsdp.wrap import CustomPolicy
import torch.nn as nn

# TODO: a more custom, easy-to-control wrapping policy, that can handle a list/dictionary of layers
# Currently implemented through a workaround that splits the custom block
class MambaLayerPolicy(CustomPolicy):
    def __init__(self, lambda_fn: Callable[[Module], Union[bool, dict[str, Any]]]):
        super().__init__(lambda_fn)
    
    def _run_policy(
        self,
        root_module: Module,
        ignored_modules: set[Module],
        root_kwargs: dict[str, Any],
    ) -> dict[Module, dict[str, Any]]:
        
        target_module_to_kwargs: dict[nn.Module, dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            res = self._lambda_fn(module)
            if not isinstance(res, (dict, bool)):
                raise ValueError(
                    "The lambda_fn passed to CustomPolicy should return "
                    f"False/True or a kwarg dict, but it returned {res}"
                )
            if not res:
                continue
            kwargs = copy.copy(root_kwargs)
            if isinstance(res, dict):
                # Override the root kwargs with the ones specified by the
                # lambda function
                kwargs.update(res)
            target_module_to_kwargs[module] = kwargs
        return target_module_to_kwargs