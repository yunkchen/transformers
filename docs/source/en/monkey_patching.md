<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Monkey patching (experimental feature)

Monkey patching allows you to replace model components globally without modifying the original model code. Once registered, patches are automatically applied when loading any model with [`~PreTrainedModel.from_pretrained`] or [`~PreTrainedModel.from_config`]. This enables you to restructure models for specific requirements like quantization compatibility, apply optimizations, or experiment with architectural variants.

> [!WARNING]
> **Monkey patching should be used as a last resort** when you need to change the layout and structure of a module and or its weights. For many customization and optimization needs, try using the [Attention interface](./attention_interface), [Experts interface](./experts_interface), or [Kernels registry](./kernel_doc/overview) instead. Only use monkey patching when you need structural changes that can't be achieved through custom forward implementations alone (e.g., for quantization library compatibility, fusing layers, or architectural experiments).

## Quick example

Here's a complete example restructuring a Mixture-of-Experts (MoE) model for quantization compatibility:

```python
from transformers import AutoModelForCausalLM, register_monkey_patch_mapping
import torch.nn as nn

class ModuleListExperts(nn.ModuleList):
    """MoE experts as a ModuleList instead of packed parameters."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(MoeMLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

# Register the patch globally - it will be automatically applied
register_monkey_patch_mapping(
    mapping={"Qwen2MoeExperts": SequentialExperts}
)

# The patch is automatically applied during loading
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
```

## Registering patches

Use [`register_monkey_patch_mapping`] to register replacements globally:

```python
from transformers import register_monkey_patch_mapping

# Register a single patch
register_monkey_patch_mapping(
    mapping={"Qwen2MoeExperts": SequentialExperts}
)

# Register multiple patches at once
register_monkey_patch_mapping(
    mapping={
        "Qwen2MoeExperts": SequentialExperts,
        "Qwen2MoeAttention": CustomAttention,
    },
    # Overwrite existing patches if they exist
    overwrite=True,
)
```

To inspect or clear patches, use [`get_monkey_patch_mapping`] and [`clear_monkey_patch_mapping`]:

```python
from transformers import get_monkey_patch_mapping, clear_monkey_patch_mapping

# Check registered patches
patches = get_monkey_patch_mapping()

# Clear all patches
clear_monkey_patch_mapping()
```

## Important notes

- **Weight handling**: Monkey patching only replaces classes, not weights. If your patched class has a different weights layout, you'll need to handle [weight conversions](./weightconverter) separately to ensure compatibility with pretrained weights.
- **Global effect**: Patches are applied globally to all models loaded after registration. Be cautious when registering patches that may affect multiple models.

## API reference

[[autodoc]] register_monkey_patch_mapping

[[autodoc]] clear_monkey_patch_mapping

[[autodoc]] get_monkey_patch_mapping

[[autodoc]] apply_monkey_patches
