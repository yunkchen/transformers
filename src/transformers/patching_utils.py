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
import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch.nn as nn

from .core_model_loading import WeightConverter


@dataclass
class PatchConfig:
    """
    Configuration for patching modeling classes during model loading.

    Args:
        mapping (`Dict[str, type[nn.Module]]`):
            A mapping from the name of the class to be patched (e.g. "Qwen2MoeExperts") to the new class that will replace it (e.g. `ModuleListExperts`).
        filtered_weight_conversion_patterns (`str` or `List[str]`, *optional*):
            A regex pattern or a list of regex patterns to filter out weight conversions.
            Any weight conversion with source or target patterns matching any of the specified patterns will be excluded from being applied during model loading.
            This can be used to prevent certain weights from being converted when the structure of the model is changed significantly due to the patching,
            and the converted weights would not be compatible with the new structure.
    """

    # Should we make the mapping from absolute module path instead of just class name to avoid potential conflicts?
    # e.g. {"transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeExperts": ModuleListExperts}
    mapping: dict[str, type[nn.Module]]
    filtered_weight_conversion_patterns: str | list[str] | None = None

    def __post_init__(self):
        if isinstance(self.filtered_weight_conversion_patterns, str):
            self.filtered_weight_conversion_patterns = [self.filtered_weight_conversion_patterns]


@contextmanager
def patching_context(model_class: type[nn.Module], patch_config: PatchConfig):
    """
    Context manager for applying temporary patches to modeling classes during model loading.
    The specified classes in patch_config.mapping will be replaced with the new classes for
    the duration of the context, and then restored to their original state afterwards.
    """

    original_classes = {}
    modeling_module = importlib.import_module(model_class.__module__)

    try:
        for module_name, replacement_class in patch_config.mapping.items():
            if hasattr(modeling_module, module_name):
                original_classes[module_name] = getattr(modeling_module, module_name)
                setattr(modeling_module, module_name, replacement_class)
            else:
                raise AttributeError(
                    f"Module '{modeling_module.__name__}' does not have a class named '{module_name}' to patch."
                )

        yield

    finally:
        for module_name, original_class in original_classes.items():
            setattr(modeling_module, module_name, original_class)


def filter_weight_conversions(
    weight_conversions: list[WeightConverter], patch_config: PatchConfig
) -> list[WeightConverter]:
    """
    Filter out weight conversions that match any of the specified source or target patterns.

    Args:
        weight_conversions (`List[WeightConverter]`):
            The list of weight conversions to filter.
        patch_config (`PatchConfig`, *optional*):
            The patch configuration containing the patterns to filter out.
    Returns:
        `List[WeightConverter]`: The filtered list of weight conversions.
    """

    if patch_config.filtered_weight_conversion_patterns is None:
        return weight_conversions

    filtered_conversions = []
    for conversion in weight_conversions:
        conversion_patterns = conversion.source_patterns + conversion.target_patterns
        if any(
            any(re.search(pattern, conv_pattern) for conv_pattern in conversion_patterns)
            for pattern in patch_config.filtered_weight_conversion_patterns
        ):
            continue

        filtered_conversions.append(conversion)

    return filtered_conversions
