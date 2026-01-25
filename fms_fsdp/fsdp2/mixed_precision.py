import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

fpSixteen = MixedPrecisionPolicy(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

bfSixteen = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
)
