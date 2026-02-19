# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal, logging, strtobool
from ..utils.quantization_config import QuantizationMethod


if TYPE_CHECKING:
    from torch import nn

logger = logging.get_logger(__name__)


def is_fsdp_managed_module(module: nn.Module) -> bool:
    if not is_torch_available():
        return False

    import torch

    if not torch.distributed.is_available():
        return False

    import torch.distributed.fsdp

    return isinstance(module, torch.distributed.fsdp.FullyShardedDataParallel) or getattr(
        module, "_is_fsdp_managed_module", False
    )


def is_fsdp_enabled():
    if is_torch_available():
        import torch

        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
            and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
        )

    return False

def initialize_fsdp(
    fsdp_plan: str | dict[str, str] | None,
    device_mesh=None,
    device_map=None,
):
    """
    Sets up the device mesh for FSDP2 (Fully Sharded Data Parallel).
    This function is called when the model is loaded and fsdp_plan is set.

    Args:
        fsdp_plan: Either "auto" for automatic sharding or a dict mapping module names to sharding strategies.
        device_mesh: Optional pre-created DeviceMesh for FSDP.
        device_map: Optional device map.

    Returns:
        Tuple of (device_map, device_mesh, fsdp_size)
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    import torch
    import torch.distributed as dist

    if fsdp_plan is None:
        return device_map, device_mesh, None

    if not is_torch_greater_or_equal("2.5"):
        raise OSError("FSDP2 is only supported for `torch>=2.5`.")

    if device_mesh is None:
        # Detect the accelerator on the machine
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            device_type = "cpu"  # fallback
        current_device = getattr(torch, device_type)

        if not dist.is_initialized():
            try:
                rank = int(os.environ["RANK"])
                local_rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ["WORLD_SIZE"])

                backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
                backend = backend_map.get(device_type)
                if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", "0")):
                    backend = "ccl"
                if device_type == "xpu" and not is_torch_greater_or_equal("2.8", accept_dev=True):
                    backend = "ccl"

                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                if device_type != "cpu":
                    current_device.set_device(local_rank)

            except Exception as e:
                raise OSError(
                    "We tried to initialize torch.distributed for you, but it failed. Make "
                    "sure you init torch distributed in your script to use `fsdp_plan`."
                ) from e

        if device_type != "cpu":
            current_device.set_device(int(os.environ["LOCAL_RANK"]))
            index = current_device.current_device()
            fsdp_device = torch.device(device_type, index)
            device_map = fsdp_device
        else:
            fsdp_device = torch.device(device_type)
            device_map = device_type or {}

        fsdp_size = dist.get_world_size()
        device_mesh = torch.distributed.init_device_mesh(fsdp_device.type, (fsdp_size,), mesh_dim_names=("dp_shard",))
    else:
        # Use provided device mesh
        if device_mesh.ndim > 1:
            if "fsdp" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `fsdp_plan` with n-d `device_mesh`, it must contain an 'fsdp' dimension. "
                    "Please provide a valid `device_mesh`."
                )
            device_mesh = device_mesh["fsdp"]
        fsdp_size = device_mesh.size()
        device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

    return device_map, device_mesh, fsdp_size


def get_transformer_block_classes(model):
    """
    Identifies transformer block classes in a model for FSDP wrapping.
    These are typically the repeated layers that benefit from FSDP sharding.

    Returns a set of module classes that should be wrapped with fully_shard().
    """
    block_classes = set()

    # Common transformer block class names
    block_names = {
        "DecoderLayer",
        "EncoderLayer",
        "TransformerBlock",
        "Block",
        "Layer",
    }

    for module in model.modules():
        class_name = module.__class__.__name__
        # Use endswith to avoid false positives (e.g. "Layer" matching "LayerNorm")
        for block_name in block_names:
            if class_name.endswith(block_name):
                block_classes.add(type(module))
                break

    return block_classes


def apply_fsdp2(
    model,
    device_mesh,
    fsdp_plan: str | dict[str, str] = "auto",
    reshard_after_forward: bool = True,
):
    """
    Apply FSDP2 (fully_shard) to a model following TorchTitan's approach.

    FSDP2 key differences from FSDP1:
    - Uses `fully_shard()` composable API instead of wrapper
    - Parameters remain as DTensors sharded on dim-0
    - No FlatParameter abstraction
    - Communication-free sharded state dicts

    Args:
        model: The model to shard with FSDP2.
        device_mesh: The DeviceMesh for FSDP (1D for pure FSDP, 2D for HSDP).
        fsdp_plan: Either "auto" for automatic block detection or a dict mapping
                   module names to strategies.
        reshard_after_forward: If True (default), parameters are resharded after forward
                               pass (ZeRO-3 style). If False, parameters stay gathered
                               (ZeRO-2 style).

    Returns:
        The FSDP-wrapped model.
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    import torch

    if not is_torch_greater_or_equal("2.5"):
        raise OSError("FSDP2 requires torch>=2.5")

    # Import FSDP2 API
    from torch.distributed._composable.fsdp import fully_shard

    if device_mesh is None:
        raise ValueError("device_mesh is required for FSDP2")

    # FSDP2 requires contiguous parameters - make them contiguous in-place
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            with torch.no_grad():
                param.data = param.data.contiguous()
            logger.debug(f"Made parameter {name} contiguous for FSDP2")

    if fsdp_plan == "auto":
        # Auto-detect transformer blocks and apply fully_shard to each
        block_classes = get_transformer_block_classes(model)

        if not block_classes:
            logger.warning(
                "Could not auto-detect transformer block classes for FSDP. "
                "Applying FSDP only to root module."
            )
        else:
            # Apply fully_shard to each transformer block (nested sharding)
            for name, module in model.named_modules():
                if type(module) in block_classes:
                    fully_shard(
                        module,
                        mesh=device_mesh,
                        reshard_after_forward=reshard_after_forward,
                    )
                    logger.debug(f"Applied fully_shard to {name} ({type(module).__name__})")

        # Apply fully_shard to the root model
        fully_shard(
            model,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
        )
        logger.info(f"FSDP2 applied to model with {len(block_classes)} block types detected")

    elif isinstance(fsdp_plan, dict):
        # Apply fully_shard based on explicit plan
        for module_pattern, strategy in fsdp_plan.items():
            for name, module in model.named_modules():
                if module_pattern in name or name == module_pattern:
                    fully_shard(
                        module,
                        mesh=device_mesh,
                        reshard_after_forward=(strategy != "no_reshard"),
                    )
                    logger.debug(f"Applied fully_shard to {name} with strategy {strategy}")

        # Always apply to root
        fully_shard(
            model,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
        )
    else:
        raise ValueError(f"fsdp_plan must be 'auto' or a dict, got {type(fsdp_plan)}")

    # Mark the model as FSDP-managed
    model._is_fsdp_managed_module = True
    model._fsdp_device_mesh = device_mesh
    model._fsdp_size = device_mesh.size()

    return model


def distribute_fsdp_model(model, fsdp_plan, device_mesh):
    """
    Distribute a model according to the FSDP plan.

    This function wraps apply_fsdp2 and sets model attributes for tracking.

    Args:
        model: The model to distribute.
        fsdp_plan: Either "auto" or a dict mapping module patterns to strategies.
        device_mesh: The DeviceMesh for FSDP communication.

    Returns:
        The FSDP-distributed model.
    """
    if fsdp_plan is None or device_mesh is None:
        return model

    model = apply_fsdp2(model, device_mesh, fsdp_plan)

    return model

# TODO(3outeille): make sure new FSDP works with PEFT
def get_fsdp_ckpt_kwargs():
    """
    Returns checkpoint kwargs for FSDP model saving.

    Checks if the `adapter_only` parameter is supported by `save_fsdp_model` from accelerate
    and returns the appropriate kwargs.
    """
    from accelerate.utils import save_fsdp_model

    if "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def update_fsdp_plugin_peft(model, accelerator):
    """
    Updates the FSDP plugin for PEFT LoRA/QLoRA compatibility.

    When using FSDP with PEFT LoRA, the auto wrap policy needs to be updated to additionally wrap
    LoRA trainable layers separately. When using FSDP with QLoRA, the mixed precision policy needs
    to be updated to use the quantization storage data type.
    """
    from peft import PeftConfig
    from peft.utils.other import fsdp_auto_wrap_policy

    if isinstance(model.active_peft_config, PeftConfig):
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    if (
        getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        and model.hf_quantizer.quantization_config.bnb_4bit_quant_storage.is_floating_point
    ):
        accelerator.state.fsdp_plugin.set_mixed_precision(
            model.hf_quantizer.quantization_config.bnb_4bit_quant_storage, override=True
        )
