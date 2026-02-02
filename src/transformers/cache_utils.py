from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch

from .configuration_utils import PreTrainedConfig
from .utils import (
    is_hqq_available,
    is_quanto_greater,
    is_torch_greater_or_equal,
    is_torchdynamo_compiling,
    logging,
)


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

_is_torch_greater_or_equal_than_2_7 = is_torch_greater_or_equal("2.7", accept_dev=True)


logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.inner: torch.Tensor | None = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, states: torch.Tensor) -> None: ...

    @abstractmethod
    def update(self, states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None) -> torch.Tensor: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            self.inner = self.inner.to("cpu", non_blocking=True)

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.is_initialized and self.inner.device != self.device:
            self.inner = self.inner.to(self.device, non_blocking=True)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.inner.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.inner = self.inner.index_select(0, beam_idx.to(self.inner.device))


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores a single tensor of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding = False

    def lazy_initialization(self, states: torch.Tensor) -> None:
        self.dtype, self.device = states.dtype, states.device
        # Create an empty tensor with the right rank to allow concatenation on the sequence dimension.
        self.inner = states.new_zeros((*states.shape[:2], 0, states.shape[-1]))
        self.is_initialized = True

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Update the cache in-place, and return the cached tensor.

        Args:
            states (`torch.Tensor`): The new states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            `torch.Tensor`: The cached states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        self.inner = torch.cat([self.inner, states], dim=-2)
        return self.inner

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.inner.numel() == 0:
            return 0
        return self.inner.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.inner = self.inner[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.inner = self.inner.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.inner = self.inner[indices, ...]


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the states as tensors of shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """

    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)

    def lazy_initialization(self, states: torch.Tensor) -> None:
        super().lazy_initialization(states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Update the cache in-place, and return the full cached states.

        Args:
            states (`torch.Tensor`): The new states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            `torch.Tensor`: The states to be consumed by the attention layer.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        self.cumulative_length += states.shape[-2]

        # Compute the full states
        full_states = torch.cat([self.inner, states], dim=-2)
        # Only cache the last `self.sliding_window - 1` tokens (or all of them if lower than that)
        self.inner = full_states[:, :, -self.sliding_window + 1 :, :]

        # Return the full states
        return full_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.inner.shape[-2]


class StaticLayer(CacheLayerMixin):
    """
    A static cache layer that stores the states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, states: torch.Tensor) -> None:
        """
        Lazy initialization of the cache tensor. This allows to get all properties (dtype, device, num_heads in case of
        TP etc...) at runtime directly, which is extremely practical as it avoids moving devices, dtypes etc later on
        for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.dtype, self.device = states.dtype, states.device
        self.max_batch_size, self.num_heads = states.shape[:2]
        self.head_dim = states.shape[-1]

        self.inner = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.inner)

        self.is_initialized = True

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Update the cache in-place, and return the cached states.

        Args:
            states (`torch.Tensor`): The new states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            `torch.Tensor`: The cached states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.inner.index_copy_(2, cache_position, states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.inner[:, :, cache_position] = states
        return self.inner

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.inner[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticSlidingWindowLayer(StaticLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, sliding_window), head_dim]`. It lazily allocates its full backing tensor,
    and then mutates it in-place. Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
        sliding_window (`int`):
            The size of the sliding window.
    """

    is_sliding = True

    def __init__(self, max_cache_len: int, sliding_window: int):
        effective_max_cache_len = min(sliding_window, max_cache_len)
        super().__init__(max_cache_len=effective_max_cache_len)
        self.cumulative_length = 0

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Update the cache in-place, and return the necessary states.

        Args:
            states (`torch.Tensor`): The new states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            `torch.Tensor`: The cached states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(states.shape[-2], device=self.device)
        )

        cumulative_length = self.cumulative_length
        is_full = cumulative_length >= self.max_cache_len
        # Update it now that we saved the value above
        self.cumulative_length += states.shape[-2]

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if states.shape[-2] == 1:
                # Roll all values to the left by 1 position
                new_inner = self.inner.roll(-1, dims=-2)
                # Overwrite the last position with new states
                # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
                index = torch.tensor([-1], dtype=int, device=self.device)
                new_inner[:, :, index] = states

                # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
                self.inner.copy_(new_inner)
                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.inner
            # Already full but using more than 1 new token (e.g. prefill caching, chat continuation, etc...)
            else:
                full_states = torch.cat((self.inner[:, :, 1:, :], states), dim=-2)
        # Not yet full, but becoming full on this update
        elif cumulative_length + states.shape[2] > self.max_cache_len:
            # Fast prefill path, no need to cat() in this case, as the cache is currently empty
            if cumulative_length == 0:
                full_states = states
            else:
                full_states = torch.cat((self.inner[:, :, :cumulative_length, :], states), dim=-2)
        else:
            try:
                self.inner.index_copy_(2, cache_position, states)
            except NotImplementedError:
                self.inner[:, :, cache_position] = states

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.inner

        # We only cache the last `sliding_window` tokens
        self.inner.copy_(full_states[:, :, -self.max_cache_len :, :])
        # we should return the whole states instead of `self.inner` here, as otherwise we lose some context
        return full_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        sliding_window = self.max_cache_len
        is_full = self.cumulative_length >= self.max_cache_len

        kv_offset = max(self.cumulative_length - sliding_window + 1, 0)
        # The cache is already full
        if is_full:
            kv_length = sliding_window + query_length - 1
        # Not yet full, but becoming full on this update
        elif self.cumulative_length + query_length > sliding_window:
            kv_length = self.cumulative_length + query_length
        # Here the Cache is still smaller than the local size, but we return the local size as it's static
        else:
            kv_length = sliding_window

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class QuantizedLayer(DynamicLayer):
    """
    A quantized layer similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for the cache by applying
    quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length`
    is set as a maximum capacity for the original precision cache. When the length goes beyond maximum capacity, the original
    precision cache is discarded and moved into the quantized cache. The quantization is done per-channel with a set `q_group_size`
    for the cached tensor, in contrast to what was described in the paper.
    """

    def __init__(
        self,
        nbits: int = 4,
        axis: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__()
        self.nbits = nbits
        self.axis = axis
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.cumulative_length = 0

    def update(
        self,
        states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Update the cache in-place, and return the necessary states.

        Args:
            states (`torch.Tensor`): The new states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            `torch.Tensor`: The states for the attention layer.
        """
        self.cumulative_length += states.shape[-2]

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(states)
            self._quantized_inner = self._quantize(states.contiguous(), axis=self.axis)
            return states

        dequant_states = self._dequantize(self._quantized_inner)
        states_to_return = torch.cat([dequant_states, self.inner, states], dim=-2)
        if self.inner.dim() == 4 and self.inner.shape[-2] + 1 >= self.residual_length:
            self._quantized_inner = self._quantize(states_to_return.contiguous(), axis=self.axis)
            self.inner = torch.tensor([], dtype=states.dtype, device=states.device)
        else:
            self.inner = torch.cat([self.inner, states], dim=-2)

        return states_to_return

    @abstractmethod
    def _quantize(self, tensor, axis): ...

    @abstractmethod
    def _dequantize(self, q_tensor): ...

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return self.cumulative_length


class QuantoQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis=axis,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        # We need to import quanto here to avoid circular imports due to optimum/quanto/models/transformers_models.py
        if is_quanto_greater("0.2.5", accept_dev=True):
            from optimum.quanto import MaxOptimizer, qint2, qint4
        else:
            raise ImportError(
                "You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedLayer`. "
            )

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis not in [0, -1]:
            raise ValueError(f"`axis` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis}")

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        from optimum.quanto import quantize_weight

        scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedLayer(QuantizedLayer):
    def __init__(
        self,
        nbits: int = 4,
        axis: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        super().__init__(
            nbits=nbits,
            axis=axis,
            q_group_size=q_group_size,
            residual_length=residual_length,
        )

        if not is_hqq_available():
            raise ImportError("You need to install `hqq` to use `HQQQuantizedLayer`")

        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis not in [0, 1]:
            raise ValueError(f"`axis` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.inner.device,
            compute_dtype=self.inner.dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.inner.dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.inner.device)  # Move to device and cast to dtype
        meta["scale"] = meta["scale"].to(qtensor.device)
        meta["zero"] = meta["zero"].to(qtensor.device)
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


class _CacheState:
    def __init__(
        self,
        layers: list[CacheLayerMixin] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin] | None = None,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a cache state either from a list `layers` of predefined `CacheLayerMixin`, or from a "
                "`layer_class_to_replicate`, in which case a new layer will be appended when needed."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a cache state."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def _ensure_layer(self, layer_idx: int):
        if self.layer_class_to_replicate is None:
            return
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

    def update(self, states: torch.Tensor, layer_idx: int, cache_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        self._ensure_layer(layer_idx)
        return self.layers[layer_idx].update(states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        if layer_idx >= len(self.layers):
            return
        if not (only_non_sliding and getattr(self.layers[layer_idx], "is_sliding", False)):
            self.layers[layer_idx].offload()

    def prefetch(self, layer_idx: int):
        if layer_idx < len(self.layers):
            self.layers[layer_idx].prefetch()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in self.layers:
            layer.batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        values = []
        for layer in self.layers:
            if hasattr(layer, "max_batch_size"):
                values.append(layer.max_batch_size)
            elif layer.is_initialized and layer.inner is not None:
                values.append(layer.inner.shape[0])
        if len(values) == 0:
            raise ValueError("No layers are initialized, cannot infer max batch size.")
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        values = []
        for layer in self.layers:
            if hasattr(layer, "max_cache_len"):
                values.append(layer.max_cache_len)
            else:
                values.append(layer.get_max_cache_shape())
        return max(values) if len(values) > 0 else 0

    @property
    def is_compileable(self) -> bool:
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    @property
    def is_sliding(self) -> list[bool]:
        return [getattr(layer, "is_sliding", False) for layer in self.layers]


class Cache:
    """
    Base cache container. Each cache groups one or more `_CacheState` objects, typically `keys` and `values`, but it
    can be extended to hold other states such as `ssm_states` or `conv_states`.

    Args:
        layers (`dict[str, list[CacheLayerMixin]]`, *optional*):
            A mapping from cache state name to list of pre-created `CacheLayerMixin`. If omitted (`None`), then
            `layer_class_to_replicate` will be used.
        layer_class_to_replicate (`type[CacheLayerMixin]` or `dict[str, type[CacheLayerMixin]]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each state
            and each layer, and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater
            than the current list of layers.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
    """

    cache_state_names: tuple[str, ...] = ("keys", "values")

    def __init__(
        self,
        layers: dict[str, list[CacheLayerMixin]] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin] | dict[str, type[CacheLayerMixin]] | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        state_names: tuple[str, ...] | None = None,
    ):
        state_names = state_names if state_names is not None else self.cache_state_names
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a mapping of predefined `layers`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append new layers on demand."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )

        self.state_names = state_names
        self.state_caches: dict[str, _CacheState] = {}

        if layers is not None:
            for name in state_names:
                if name not in layers:
                    raise ValueError(f"Missing layers definition for state `{name}`.")
                self.state_caches[name] = _CacheState(layers=layers[name])
        else:
            for name in state_names:
                if isinstance(layer_class_to_replicate, dict):
                    if name not in layer_class_to_replicate:
                        raise ValueError(f"Missing `layer_class_to_replicate` entry for state `{name}`.")
                    cache_layer_class = layer_class_to_replicate[name]
                else:
                    cache_layer_class = layer_class_to_replicate
                self.state_caches[name] = _CacheState(layer_class_to_replicate=cache_layer_class)

        for name, cache_state in self.state_caches.items():
            setattr(self, name, cache_state)

        self.offloading = offloading
        if self.offloading:
            self.only_non_sliding = offload_only_non_sliding
            self.prefetch_stream = torch.Stream() if _is_torch_greater_or_equal_than_2_7 else torch.cuda.Stream()

    def __repr__(self):
        return f"{self.__class__.__name__}(states={list(self.state_caches.keys())})"

    @property
    def layers(self) -> list[CacheLayerMixin]:
        """Primary cache layers (kept for backwards compatibility)."""
        return self.state_caches[self.state_names[0]].layers

    def _primary_cache(self) -> _CacheState:
        return self.state_caches[self.state_names[0]]

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Prefetch a given layer on its device. If `only_non_sliding` is True, it will try to prefetch only the layers
        which are non-sliding. If the `layer_idx` is outside the range, this will circle back to the first layers.
        Note that we use a non-default stream for this, to avoid blocking.
        """
        if len(self.layers) == 0:
            return
        if only_non_sliding:
            # Try to find next non-sliding, starting at `layer_idx`
            try:
                layer_idx = layer_idx + self.is_sliding[layer_idx:].index(False)
            # In this case, we need to circle back to the beginning
            except ValueError:
                layer_idx = self.is_sliding.index(False)
        else:
            layer_idx = layer_idx if layer_idx < len(self.layers) else 0

        with self.prefetch_stream if _is_torch_greater_or_equal_than_2_7 else torch.cuda.stream(self.prefetch_stream):
            for cache_state in self.state_caches.values():
                cache_state.prefetch(layer_idx)

    def offload(self, layer_idx: int, only_non_sliding: bool = True):
        """
        Offload a given `layer_idx`. If `only_non_sliding` is True, it will offload `layer_idx` only if it is a
        non-sliding layer. Note that we do it on the default stream, so that we ensure all earlier
        computation in the layer's `update` methods are finished.
        """
        if len(self.layers) == 0:
            return
        for cache_state in self.state_caches.values():
            cache_state.offload(layer_idx, only_non_sliding)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        if len(self.state_names) < 2:
            raise ValueError("Cache.update expects at least two cache states.")
        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        key_cache, value_cache = (self.state_caches[self.state_names[0]], self.state_caches[self.state_names[1]])
        keys = key_cache.update(key_states, layer_idx, cache_kwargs)
        values = value_cache.update(value_states, layer_idx, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values

    def early_initialization(
        self, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        fake_tensor = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
        for cache_state in self.state_caches.values():
            for layer in cache_state.layers:
                layer.lazy_initialization(fake_tensor)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        primary_cache = self._primary_cache()
        reference_length = primary_cache.get_seq_length(layer_idx)
        for name, cache_state in self.state_caches.items():
            if cache_state is primary_cache:
                continue
            if cache_state.get_seq_length(layer_idx) not in (reference_length, 0):
                raise ValueError(f"Cache `{name}` has a different sequence length than the primary cache.")
        return reference_length

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        return self._primary_cache().get_mask_sizes(cache_position, layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        return self._primary_cache().get_max_cache_shape(layer_idx)

    def reset(self):
        """Recursively reset all layers tensors"""
        for cache_state in self.state_caches.values():
            cache_state.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search"""
        for cache_state in self.state_caches.values():
            cache_state.reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for cache_state in self.state_caches.values():
            cache_state.crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for cache_state in self.state_caches.values():
            cache_state.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Select indices from the cache"""
        for cache_state in self.state_caches.values():
            cache_state.batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = []
        for cache_state in self.state_caches.values():
            if len(cache_state.layers) > 0:
                values.append(cache_state.max_batch_size)
        if len(values) == 0:
            raise ValueError("Cannot infer max batch size without initialized layers.")
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across states: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [cache_state.max_cache_len for cache_state in self.state_caches.values()]
        return max(values) if len(values) > 0 else 0

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        if any(len(state.layers) == 0 for state in self.state_caches.values()):
            return False
        return all(state.is_compileable for state in self.state_caches.values())

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized"""
        return len(self.layers) > 0 and all(state.is_initialized for state in self.state_caches.values())

    @property
    def is_sliding(self) -> list[bool]:
        """Return whether the layers of the cache are sliding window"""
        sliding = None
        for cache_state in self.state_caches.values():
            if sliding is None:
                sliding = cache_state.is_sliding
            elif sliding != cache_state.is_sliding:
                raise ValueError("Sliding configuration is not consistent across cache states.")
        return sliding or []

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        return len(self.layers)


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as two lists of `CacheLayer`, one for each layer. The expected shape for each
    tensor in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        key_layers = []
        value_layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    key_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
                    value_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
                else:
                    key_layer = DynamicLayer()
                    value_layer = DynamicLayer()
                key_layers.append(key_layer)
                value_layers.append(value_layer)

        # In this case, use the passed data to already fill in the Cache
        if ddp_cache_data is not None:
            # Init all the layers with the data
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                # If the config was not passed above, initialize a new cache layer for each entry of the ddp_data
                if config is None:
                    # kv_and_optional_sliding contains at least two elements: the key and value states. It can also
                    # contain a third element, which is an optional sliding window tensor.
                    sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                    # If there is a sliding window tensor, use it to initialize the layer
                    if sliding_window_tensor is not None:
                        # Since the same layer is dispatched across replicas, sliding_window is the same for all
                        sliding_window = sliding_window_tensor[0].item()
                        key_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
                        value_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
                    else:
                        key_layer = DynamicLayer()
                        value_layer = DynamicLayer()
                    key_layers.append(key_layer)
                    value_layers.append(value_layer)
                # Update the layer with the data
                key_layers[layer_idx].update(kv_and_optional_sliding[0], cache_kwargs=None)
                value_layers[layer_idx].update(kv_and_optional_sliding[1], cache_kwargs=None)

        # If neither of config nor ddp_data was passed, then simply lazy init a full cache of DynamicLayer
        if len(key_layers) == 0:
            super().__init__(
                layer_class_to_replicate={"keys": DynamicLayer, "values": DynamicLayer},
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(
                layers={"keys": key_layers, "values": value_layers},
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )

    def __iter__(self):
        for key_layer, value_layer in zip(self.keys.layers, self.values.layers):
            yield key_layer.inner, value_layer.inner, getattr(key_layer, "_sliding_window_tensor", None)


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
    for potential hybrid cache structure, and initialize each layer accordingly.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used. It will be used to check for sliding
            or hybrid layer structure, and initialize each layer accordingly.
        max_cache_len (`int`):
            The maximum number of tokens that this Cache should hold.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
    def __init__(
        self,
        config: PreTrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
            else:
                layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        key_layers = []
        value_layers = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                key_layer = StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
                value_layer = StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window)
            elif layer_type == "chunked_attention":
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                key_layer = StaticSlidingWindowLayer(
                    max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size
                )
                value_layer = StaticSlidingWindowLayer(
                    max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size
                )
            else:
                key_layer = StaticLayer(max_cache_len=max_cache_len)
                value_layer = StaticLayer(max_cache_len=max_cache_len)
            key_layers.append(key_layer)
            value_layers.append(value_layer)

        super().__init__(
            layers={"keys": key_layers, "values": value_layers},
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )


class QuantizedCache(Cache):
    """
    A quantizer cache similar to what is described in the
    [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://huggingface.co/papers/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for keys and values
    by applying quantization.
    The cache has two types of storage, one for original precision and one for the
    quantized cache. A `residual length` is set as a maximum capacity for the original precision cache. When the
    length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache.
    The quantization is done per-channel with a set `q_group_size` for both keys and values, in contrast to what was
    described in the paper.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        backend (`str`):
            The quantization backend to use. One of `("quanto", "hqq").
        config (`PreTrainedConfig`):
            The config of the model for which this Cache will be used.
        nbits (`int`, *optional*, defaults to 4):
            The number of bits for quantization.
        axis_key (`int`, *optional*, defaults to 0):
            The axis on which to quantize the keys.
        axis_value (`int`, *optional*, defaults to 0):
            The axis on which to quantize the values.
        q_group_size (`int`, *optional*, defaults to 64):
            Quantization is done per-channel according to a set `q_group_size` for both keys and values.
        residual_length (`int`, *optional*, defaults to 128):
            Maximum capacity for the original precision cache
    """

    def __init__(
        self,
        backend: str,
        config: PreTrainedConfig,
        nbits: int = 4,
        axis_key: int = 0,
        axis_value: int = 0,
        q_group_size: int = 64,
        residual_length: int = 128,
    ):
        if backend == "quanto":
            layer_class = QuantoQuantizedLayer
        elif backend == "hqq":
            layer_class = HQQQuantizedLayer
        else:
            raise ValueError(f"Unknown quantization backend `{backend}`")

        config = config.get_text_config(decoder=True)
        key_layers = [
            layer_class(nbits, axis_key, q_group_size, residual_length) for _ in range(config.num_hidden_layers)
        ]
        value_layers = [
            layer_class(nbits, axis_value, q_group_size, residual_length) for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers={"keys": key_layers, "values": value_layers})


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        caches (`Iterable`):
            Usually an iterable of length 2, containing 2 `Cache` objects, the first one for self-attention, the
            second one for cross-attention. Can optionally also be an iterable of length 1, containing a
            `tuple[tuple[torch.Tensor]]` (usually used for compatibility with torch dp and ddp).

    Example:

    ```python
    >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

    >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
    >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

    >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

    >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
    >>> self_attention_cache = DynamicCache(config=self.config)
    >>> cross_attention_cache = DynamicCache(config=self.config)
    >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    EncoderDecoderCache()
    ```
    """

    def __init__(self, *caches) -> None:
        # For dp and ddp support, if only one argument is passed, it should be an iterable of DynamicCache ddp data
        if len(caches) == 1:
            self_attention_cache_data, cross_attention_cache_data = [], []
            for combined_cache_data in caches[0]:
                if len(combined_cache_data) == 6:  # two tuple of style (self_attn_k, self_attn_v, self_attn_sliding)
                    self_attention_cache_data.append(combined_cache_data[:3])
                    cross_attention_cache_data.append(combined_cache_data[3:])
                # To support old DDP-style init, we handle the case where the tuple has no sliding window tensor
                elif len(combined_cache_data) == 4:  # two tuple of style (self_attn_k, self_attn_v)
                    self_attention_cache_data.append(combined_cache_data[:2])
                    cross_attention_cache_data.append(combined_cache_data[2:])
                else:
                    raise ValueError(f"Expected {len(combined_cache_data) = } to be 4 or 6.\n{combined_cache_data = }")
            self.self_attention_cache = DynamicCache(self_attention_cache_data)
            self.cross_attention_cache = DynamicCache(cross_attention_cache_data)
        # Otherwise, we should get two arguments, a self-attention cache and a cross-attention cache
        elif len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(f"One of the two arguments is not a Cache: {type(caches[0]) = }, {type(caches[1]) = }")
            self.self_attention_cache = caches[0]
            self.cross_attention_cache = caches[1]
        # Error case
        else:
            raise ValueError(f"Expected 1 or 2 arguments, got {len(caches)}")

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __iter__(self):
        """Returns tuples of style (self_attn_k, self_attn_v, self_attn_sliding, cross_attn_k, cross_attn_v, cross_attn_sliding)"""
        for self_attention_layer, cross_attention_layer in zip(self.self_attention_cache, self.cross_attention_cache):
            yield self_attention_layer + cross_attention_layer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(self_attention_cache={self.self_attention_cache}, cross_attention_cache="
            f"{self.cross_attention_cache})"
        )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_values` length, e.g. `len(past_key_values)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise TypeError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """
        Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search (on the Hub).
        """
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "list[EncoderDecoderCache]":
        """
        Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`
        """
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search (on the Hub)."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        return self.self_attention_cache.get_max_cache_shape()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.self_attention_cache.get_mask_sizes(cache_position, layer_idx)

    @property
    def is_sliding(self):
        return self.self_attention_cache.is_sliding

    @property
    def is_compileable(self) -> bool:
        return self.self_attention_cache.is_compileable
