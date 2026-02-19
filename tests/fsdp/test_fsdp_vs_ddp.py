# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""
Tests for FSDP2 vs DDP correctness.
Run:
    pytest tests/fsdp/test_fsdp_vs_ddp.py -v -s
"""

import logging
import os

import pytest

from transformers import AutoConfig, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import (
    Colors,
    backend_device_count,
    get_torch_dist_unique_port,
    init_test_logger,
    require_fsdp,
    require_torch_multi_accelerator,
    torch_device,
)
from transformers.trainer_utils import set_seed


logger = logging.getLogger("transformers.training_test")


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.tensor import DTensor
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.integrations.fsdp import apply_fsdp2, initialize_fsdp


# =============================================================================
# Distributed helper functions (following test_fsdp2.py pattern exactly)
# =============================================================================


def global_wrapper(rank, func, world_size, port, func_args, func_kwargs):
    """Set up distributed environment and run the test function."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # NOTE(3outeille): tells cuBLAS to use a deterministic workspace of 4096 entries × 8 bytes.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def init_distributed(world_size: int):
    """Decorator to run function in distributed mode using mp.spawn."""

    def _init_distributed(func):
        def wrapper(*args, **kwargs):
            port = get_torch_dist_unique_port()
            spawn_args = (func, world_size, port, args, kwargs)
            mp.spawn(global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed


def skip_if_insufficient_devices(nproc_per_node):
    """Skip test if not enough GPUs available."""
    if backend_device_count(torch_device) < nproc_per_node:
        pytest.skip(f"Need at least {nproc_per_node} devices, have {backend_device_count(torch_device)}")


# =============================================================================
# Utility functions
# =============================================================================

#TODO(3outeille): run slow model?
MODEL_NAME = "JackFram/llama-160m"
# MODEL_NAME = "Corianas/Tiny-Moe"
BATCH_SIZE = 2
SEQ_LEN = 1024
NUM_STEPS = 20
LR = 3e-4
SEED = 42

def log_comparison_table(title, ddp_vals, fsdp_vals):
    """Log a side-by-side comparison table for DDP vs FSDP2 values."""
    C = Colors
    SEP = f"{C.DIM}|{C.RESET}"
    ROW = f"  {C.DIM}{'─' * 52}{C.RESET}"

    logger.info(f"  {C.BOLD}{title}{C.RESET}")
    logger.info(ROW)
    logger.info(
        f"  {C.DIM}{'step':>4}{C.RESET}  "
        f"{SEP}  {C.BLUE}{C.BOLD}{'DDP':^14}{C.RESET}  "
        f"{SEP}  {C.MAGENTA}{C.BOLD}{'FSDP2':^14}{C.RESET}  "
        f"{SEP}  {C.DIM}{'diff':^10}{C.RESET}"
    )
    logger.info(ROW)
    for step in range(len(ddp_vals)):
        diff = abs(ddp_vals[step] - fsdp_vals[step])
        match = f"{C.GREEN}={C.RESET}" if diff < 1e-6 else f"{C.YELLOW}{diff:.1e}{C.RESET}"
        logger.info(
            f"  {C.DIM}{step + 1:>4}{C.RESET}  "
            f"{SEP}  {C.BLUE}{ddp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {C.MAGENTA}{fsdp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {match:^10}"
        )
    logger.info(ROW)

def create_deterministic_data(batch_size, seq_len, vocab_size, device, seed):
    """Create deterministic random training data using torch.randint."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)]

def gather_fsdp2_state_dict(model):
    """Gather FSDP2 sharded parameters into full tensors via DTensor.full_tensor()."""
    state_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            state_dict[name] = param.full_tensor().clone().detach()
        else:
            state_dict[name] = param.clone().detach()
    for name, buf in model.named_buffers():
        if isinstance(buf, DTensor):
            state_dict[name] = buf.full_tensor().clone().detach()
        else:
            state_dict[name] = buf.clone().detach()
    return state_dict


def compute_grad_norm(model):
    """Compute total gradient L2 norm, gathering DTensor shards to full tensors.

    clip_grad_norm_ doesn't compute correct global norms for FSDP2's sharded DTensor
    parameters (it only sees the local shard). This function gathers full gradients
    before computing the norm.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            total_norm_sq += grad.data.float().norm(2).item() ** 2
    return total_norm_sq ** 0.5


def train_ddp(rank, config, batches, lr, device, dtype):
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    ddp_model = DDP(model, device_ids=[rank]).to(dtype)
    ddp_model.train()

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = ddp_model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = compute_grad_norm(ddp_model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = {k: v.clone().detach() for k, v in ddp_model.module.state_dict().items()}
    return losses, grad_norms, state_dict


def train_fsdp2(rank, config, batches, lr, device_map, device_mesh, dtype):
    """Run an FSDP2 training loop with Adam.

    Returns (losses, grad_norms, state_dict).
    """
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = gather_fsdp2_state_dict(model)
    return losses, grad_norms, state_dict

def _test_fsdp2_vs_ddp_impl(rank, dtype):
    """Compare losses, grad norms, and final weights between DDP and FSDP2."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(MODEL_NAME)

    batches = create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, config, batches, LR, device, dtype)

    dist.barrier()

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
        rank, config, batches, LR, device_map, device_mesh, dtype
    )

    dist.barrier()

    if rank == 0:
        logger.info("")
        log_comparison_table("Loss per step", ddp_losses, fsdp_losses)
        logger.info("")
        log_comparison_table("Gradient norm per step", ddp_grad_norms, fsdp_grad_norms)
        logger.info("")

    # Compare per-step losses
    for step in range(NUM_STEPS):
        torch.testing.assert_close(
            torch.tensor(ddp_losses[step]),
            torch.tensor(fsdp_losses[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} loss mismatch: DDP={ddp_losses[step]}, FSDP2={fsdp_losses[step]}",
        )

    # Compare per-step gradient norms
    for step in range(NUM_STEPS):
        torch.testing.assert_close(
            torch.tensor(ddp_grad_norms[step]),
            torch.tensor(fsdp_grad_norms[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} grad norm mismatch: DDP={ddp_grad_norms[step]}, FSDP2={fsdp_grad_norms[step]}",
        )

    # Compare final weights
    for key in ddp_state_dict:
        assert key in fsdp_state_dict, f"Key {key} missing from FSDP2 state dict"
        torch.testing.assert_close(
            ddp_state_dict[key],
            fsdp_state_dict[key],
            rtol=1e-4,
            atol=1e-4,
            msg=f"Weight mismatch for {key}",
        )

@pytest.mark.parametrize("nproc_per_node", [pytest.param(2, id="2gpus")])
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float32, id="float32"), pytest.param(torch.bfloat16, id="bfloat16")],
)
@require_fsdp
@require_torch_multi_accelerator
def test_fsdp2_vs_ddp(nproc_per_node, dtype):
    """20-step Adam: compare per-step losses, grad norms, and final weights."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(world_size=nproc_per_node)(_test_fsdp2_vs_ddp_impl)(dtype)
