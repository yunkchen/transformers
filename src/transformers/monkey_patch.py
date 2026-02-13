# Copyright 2026 The HuggingFace Inc. team.
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

import importlib
from contextlib import contextmanager

import torch.nn as nn

from .utils import logging


logger = logging.get_logger(__name__)

__all__ = [
    "register_monkey_patch_mapping",
    "clear_monkey_patch_mapping",
    "get_monkey_patch_mapping",
    "apply_monkey_patches",
]

_monkey_patch_mapping_cache: dict[str, type[nn.Module]] = {}


def register_monkey_patch_mapping(mapping: dict[str, type[nn.Module]], overwrite: bool = False) -> None:
    """
    Register class mappings to enable automatic patching during model creation (from_pretrained and from_config).

    Use this to register class replacements that will be automatically applied when loading any model.
    This is useful for quantization library compatibility, structural optimizations, and architectural
    experimentation. The mapping is global, can grow with multiple calls, and can be cleared entirely.

    Args:
        mapping (`Dict[str, type[nn.Module]]`):
            Mapping from original class names to replacement classes. Class names must exactly match
            those in the model's module (e.g., `"Qwen2MoeExperts"` → `CustomExperts`).
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to overwrite existing mappings for class names that are already registered.

    Example:
        ```python
        from transformers import AutoModelForCausalLM, register_monkey_patch_mapping

        # Define custom expert implementation
        class SequentialExperts(nn.Module):
            ...

        # Register the patch globally
        register_monkey_patch_mapping(
            mapping={"Qwen2MoeExperts": SequentialExperts}
        )

        # The patch will be automatically applied during loading
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-57B-A14B-Instruct")

        # You can register more patches later
        register_monkey_patch_mapping(
            mapping={"LlamaAttention": CustomAttention}
        )
        ```

    Note:
        For weight conversions, use [`~transformers.register_checkpoint_conversion_mapping`] instead.
    """
    global _monkey_patch_mapping_cache
    for class_name, replacement_class in mapping.items():
        if class_name in _monkey_patch_mapping_cache and not overwrite:
            raise ValueError(
                f"Class '{class_name}' already has a patch mapping registered. Use overwrite=True to replace it."
            )
        _monkey_patch_mapping_cache[class_name] = replacement_class


def get_monkey_patch_mapping() -> dict[str, type[nn.Module]]:
    """
    Get all registered patch mappings.

    Returns:
        `Dict[str, type[nn.Module]]`: The global class mapping dictionary.
    """
    return _monkey_patch_mapping_cache


def clear_monkey_patch_mapping() -> None:
    """
    Clear all registered patch mappings.

    This removes all registered class replacements from the global registry.

    Example:
        ```python
        from transformers import register_monkey_patch_mapping, clear_monkey_patch_mapping

        # Register some patches
        register_monkey_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Clear all patches
        clear_monkey_patch_mapping()
        ```
    """
    global _monkey_patch_mapping_cache
    _monkey_patch_mapping_cache.clear()


@contextmanager
def apply_monkey_patches(model_class: type[nn.Module]):
    """
    Context manager to temporarily apply all registered patches.

    This replaces specified classes in the model's module with registered replacements for the
    duration of the context, then restores them to their original state.

    Args:
        model_class (`type[nn.Module]`):
            The model class to patch (e.g., `Qwen2MoeForCausalLM`).

    Example:
        ```python
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

        with apply_monkey_patches(Qwen2MoeForCausalLM):
            # Model classes are patched here
            model = Qwen2MoeForCausalLM(config)
        # Original classes are restored
        ```
    """
    mapping = get_monkey_patch_mapping()
    if not mapping:
        yield
        return

    modeling_module = importlib.import_module(model_class.__module__)
    original_classes = {}

    try:
        for module_name, replacement_class in mapping.items():
            if hasattr(modeling_module, module_name):
                original_classes[module_name] = getattr(modeling_module, module_name)
                setattr(modeling_module, module_name, replacement_class)
            else:
                logger.debug(f"Skipping patch for '{module_name}': not found in module '{modeling_module.__name__}'.")

        yield

    finally:
        for module_name, original_class in original_classes.items():
            setattr(modeling_module, module_name, original_class)
