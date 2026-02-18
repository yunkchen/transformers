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

import sys
import threading
from contextlib import contextmanager

from .utils import is_torch_available, logging
from .utils.output_capturing import OutputRecorder


if is_torch_available():
    import torch.nn as nn

logger = logging.get_logger(__name__)


_monkey_patch_mapping_cache: dict[str, type[nn.Module]] = {}
_monkey_patch_lock = threading.Lock()


def register_monkey_patch_mapping(mapping: dict[str, type[nn.Module]], overwrite: bool = False) -> None:
    """
    Register class mappings to enable automatic patching during model creation using `from_pretrained`,
    `from_config` or within the `apply_monkey_patches` context manager.

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
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patch import register_monkey_patch_mapping

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
    with _monkey_patch_lock:
        for class_name, replacement_class in mapping.items():
            # Validate that replacement_class is actually a class and is a subclass of nn.Module
            if not isinstance(replacement_class, type):
                raise TypeError(
                    f"Replacement for '{class_name}' must be a class, got {type(replacement_class).__name__}"
                )
            if not issubclass(replacement_class, nn.Module):
                raise TypeError(
                    f"Replacement class for '{class_name}' must be a subclass of nn.Module, "
                    f"got {replacement_class.__name__} which inherits from {[c.__name__ for c in replacement_class.__mro__[1:]]}"
                )

            if class_name in _monkey_patch_mapping_cache and not overwrite:
                raise ValueError(
                    f"Class '{class_name}' already has a patch mapping registered. Use overwrite=True to replace it."
                )
            _monkey_patch_mapping_cache[class_name] = replacement_class


def unregister_monkey_patch_mapping(class_names: list[str]) -> None:
    """
    Unregister class mappings to disable automatic patching for specified classes.

    This removes specified class replacements from the global registry, preventing them from being applied
    during model loading.

    Args:
        class_names (`List[str]`):
            List of original class names to remove from the patch mapping (e.g., `["Qwen2MoeExperts"]`).

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patch import register_monkey_patch_mapping, unregister_monkey_patch_mapping

        # Register a patch
        register_monkey_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Unregister the patch
        unregister_monkey_patch_mapping(["Qwen2MoeExperts"])

        # The patch will no longer be applied during loading
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
        ```
    """
    global _monkey_patch_mapping_cache
    with _monkey_patch_lock:
        for class_name in class_names:
            if class_name in _monkey_patch_mapping_cache:
                del _monkey_patch_mapping_cache[class_name]
            else:
                logger.debug(f"Class '{class_name}' not found in monkey patch mapping cache. Skipping unregistration.")


def get_monkey_patch_mapping() -> dict[str, type[nn.Module]]:
    """
    Get all registered patch mappings.

    Returns:
        `Dict[str, type[nn.Module]]`: The global class mapping dictionary.
    """
    with _monkey_patch_lock:
        return _monkey_patch_mapping_cache.copy()


def clear_monkey_patch_mapping() -> None:
    """
    Clear all registered patch mappings.

    This removes all registered class replacements from the global registry.

    Example:
        ```python
        from transformers.monkey_patch import register_monkey_patch_mapping, clear_monkey_patch_mapping

        # Register some patches
        register_monkey_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Clear all patches
        clear_monkey_patch_mapping()
        ```
    """
    global _monkey_patch_mapping_cache
    with _monkey_patch_lock:
        _monkey_patch_mapping_cache.clear()


@contextmanager
def apply_monkey_patches():
    """
    Context manager to apply registered monkey patches within a block of code.

    This temporarily replaces original classes with their registered replacements during the execution of the block, and restores the original classes afterward.

    Example:
        ```python
        from transformers import Qwen2MoeModel, Qwen2MoeConfig
        from transformers.monkey_patch import register_monkey_patch_mapping, apply_monkey_patches

        # Register a patch
        register_monkey_patch_mapping(
            mapping={"Qwen2MoeExperts": CustomExperts}
        )

        # Apply patches within the context
        with apply_monkey_patches():
            # The model will use CustomExperts instead of Qwen2MoeExperts
            model = Qwen2MoeModel(Qwen2MoeConfig())

        # Outside the context, original classes are restored
        # The model will use Qwen2MoeExperts again
        model = Qwen2MoeModel(Qwen2MoeConfig())
        ```
    """
    mapping = get_monkey_patch_mapping()
    if not mapping:
        yield
        return

    original_classes = {}
    for class_name, replacement_class in mapping.items():
        # Create list to avoid dict changed during iteration
        for module in list(sys.modules.values()):
            if module is not None and hasattr(module, "__name__"):
                if module.__name__.startswith("transformers") and hasattr(module, class_name):
                    original_class = getattr(module, class_name)
                    original_classes[(module.__name__, class_name)] = original_class
                    setattr(module, class_name, replacement_class)

    yield

    for (module_name, class_name), original_class in original_classes.items():
        module = sys.modules[module_name]
        setattr(module, class_name, original_class)


# _can_record_outputs is a class attribute so patching and unpatching it in the class won't work
# since the model instance will still reference the original class's _can_record_outputs.
def patch_output_recorders(model: nn.Module) -> None:
    """
    Patch the model instance's output recorders to use the registered replacement classes.

    This function updates output recorders in a model's submodules to use monkey-patched replacement
    classes. Output recorders are used by the transformers library to track intermediate outputs during
    forward passes (via the `_can_record_outputs` attribute). When classes are monkey-patched, these
    recorders need to be updated to reference the new classes.

    This is automatically called during model initialization when loading with `from_pretrained` or
    `from_config`. You typically don't need to call this manually unless you're constructing models
    in custom ways.

    Note:
        The `_can_record_outputs` attribute is a class-level attribute that maps output names to either:
        - `OutputRecorder` instances that have a `target_class` attribute
        - Class types directly

        This function patches both cases to use the replacement classes from the monkey patch registry.

    Args:
        model (`nn.Module`):
            The model instance whose output recorders should be patched. All submodules will be
            traversed to find and patch their `_can_record_outputs` attributes.

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from transformers.monkey_patch import register_monkey_patch_mapping, patch_output_recorders

        # Register a patch
        register_monkey_patch_mapping(mapping={"Qwen2MoeExperts": CustomExperts})

        # If you construct a model manually (without from_pretrained), patch recorders
        model = Qwen2MoeModel(config)
        patch_output_recorders(model)  # Updates output recorders to use CustomExperts
        ```
    """

    mapping = get_monkey_patch_mapping()
    if not mapping:
        return

    for class_name, replacement_class in mapping.items():
        for submodule in model.modules():
            if hasattr(submodule, "_can_record_outputs"):
                for output, recorder in submodule._can_record_outputs.items():
                    if isinstance(recorder, OutputRecorder) and recorder.target_class.__name__ == class_name:
                        recorder.target_class = replacement_class
                    elif isinstance(recorder, type) and recorder.__name__ == class_name:
                        submodule._can_record_outputs[output] = replacement_class
