"""An independently stored EMA teacher for the FSDP student.

Only local parameter shards are stored, in FP32 on CPU. Teacher inference
temporarily lends those values to the existing rollout synchronization path;
the student's parameters (and optimizer identities) are always restored. The
fixed KL reference model is never involved. Checkpoints require the same FSDP
world size and wrapping/layout on resume.
"""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch


def ema_beta(step, beta_start=0.99, beta_end=0.9999, total_steps=100):
    """Cosine teacher retention: step 1=start, step T=end, clamped thereafter."""
    if not 0 <= beta_start <= beta_end < 1:
        raise ValueError("EMA retention must satisfy 0 <= beta_start <= beta_end < 1")
    if total_steps < 1 or step < 1:
        raise ValueError("EMA step and schedule horizon must be positive")
    progress = min((int(step) - 1) / max(int(total_steps) - 1, 1), 1.0)
    if total_steps == 1:
        progress = 1.0
    return float(beta_end - (beta_end - beta_start) * (1 + math.cos(math.pi * progress)) / 2)


class ShardedEMATeacher:
    """CPU EMA state with no Parameter, optimizer, or autograd graph of its own."""

    format_version = 1

    def __init__(self, module, *, beta_start=0.99, beta_end=0.9999, total_steps=100):
        self.module = module
        self.schedule = dict(beta_start=float(beta_start), beta_end=float(beta_end), total_steps=int(total_steps))
        ema_beta(1, **self.schedule)  # validate before allocating the snapshot
        self.last_step = 0
        self.last_beta = None
        self.rollout_rng_state = None
        self.in_teacher_context = False
        self.shards = {
            name: tensor.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, tensor in module.named_parameters()
        }
        if not self.shards:
            raise ValueError("Cannot initialize an EMA teacher without student parameters")
        # Non-parameter buffers (e.g. rotary frequencies) are copied, never EMA'd.
        # Include non-persistent buffers: state_dict() alone omits those.
        self.buffers = {
            name: tensor.detach().cpu().clone() for name, tensor in module.named_buffers()
        }

    def _student_tensors(self):
        parameters = dict(self.module.named_parameters())
        buffers = dict(self.module.named_buffers())
        self._check_layout(parameters, self.shards, "parameter")
        self._check_layout(buffers, self.buffers, "buffer")
        for name, parameter in parameters.items():
            local_shard = getattr(parameter, "_local_shard", None)
            if local_shard is not None and (
                local_shard.shape != parameter.shape or local_shard.data_ptr() != parameter.data_ptr()
            ):
                raise RuntimeError(f"EMA requires idle, resharded FSDP parameters: {name}")
        return parameters, buffers

    @staticmethod
    def _check_layout(actual, expected, kind):
        if list(actual) != list(expected):
            raise ValueError(f"EMA {kind} names/order differ from the saved FSDP layout")
        for name, tensor in actual.items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"EMA {kind} shard shape mismatch for {name}")

    @torch.no_grad()
    def update(self, step):
        if self.in_teacher_context:
            raise RuntimeError("Cannot update EMA while teacher weights are active")
        if int(step) != self.last_step + 1:
            raise ValueError(f"EMA must update exactly once per outer batch: last={self.last_step}, next={step}")
        parameters, buffers = self._student_tensors()
        beta = ema_beta(int(step), **self.schedule)
        for name, parameter in parameters.items():
            self.shards[name].mul_(beta).add_(parameter.detach().to(device="cpu", dtype=torch.float32), alpha=1-beta)
        for name, buffer in buffers.items():
            self.buffers[name].copy_(buffer.detach().cpu())
        self.last_step = int(step)
        self.last_beta = beta
        return beta

    @contextmanager
    def use_for_rollout(self):
        """Swap values, not Parameter objects; restore even if generation fails."""
        if self.in_teacher_context:
            raise RuntimeError("Nested EMA teacher inference is not supported")
        parameters, buffers = self._student_tensors()
        student = {name: tensor.detach().cpu().clone() for name, tensor in parameters.items()}
        student_buffers = {name: tensor.detach().cpu().clone() for name, tensor in buffers.items()}
        self.in_teacher_context = True
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self.shards[name].to(device=parameter.device, dtype=parameter.dtype))
                for name, buffer in buffers.items():
                    buffer.copy_(self.buffers[name].to(device=buffer.device, dtype=buffer.dtype))
            yield
        finally:
            # The rollout path may have offloaded the FSDP module to CPU. Copy to
            # the CURRENT device, preserving both FSDP storage and optimizer ids.
            with torch.no_grad():
                restored_parameters, restored_buffers = self._student_tensors()
                for name, parameter in restored_parameters.items():
                    parameter.copy_(student[name].to(device=parameter.device, dtype=parameter.dtype))
                for name, buffer in restored_buffers.items():
                    buffer.copy_(student_buffers[name].to(device=buffer.device, dtype=buffer.dtype))
            self.in_teacher_context = False

    def state_dict(self, *, rank, world_size):
        if self.in_teacher_context:
            raise RuntimeError("Cannot checkpoint while teacher weights are active")
        return dict(format_version=self.format_version, rank=int(rank), world_size=int(world_size),
                    schedule=dict(self.schedule), last_step=self.last_step, last_beta=self.last_beta,
                    parameters=self.shards, buffers=self.buffers, rollout_rng_state=self.rollout_rng_state)

    def load_state_dict(self, state, *, rank, world_size, expected_step):
        if self.in_teacher_context:
            raise RuntimeError("Cannot restore while teacher weights are active")
        expected = dict(format_version=self.format_version, rank=int(rank), world_size=int(world_size),
                        schedule=self.schedule, last_step=int(expected_step))
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"EMA checkpoint {key} mismatch: expected {value}, got {state.get(key)}")
        wanted_beta = ema_beta(expected_step, **self.schedule) if expected_step else None
        if state.get("last_beta") != wanted_beta:
            raise ValueError("EMA checkpoint beta does not match its schedule and step")
        self._check_layout(state["parameters"], self.shards, "parameter")
        self._check_layout(state["buffers"], self.buffers, "buffer")
        self.shards = {name: value.detach().to(device="cpu", dtype=torch.float32).clone()
                       for name, value in state["parameters"].items()}
        self.buffers = {name: value.detach().cpu().clone() for name, value in state["buffers"].items()}
        self.last_step = int(expected_step)
        self.last_beta = wanted_beta
        rng_state = state.get("rollout_rng_state")
        self.rollout_rng_state = rng_state.detach().cpu().clone() if rng_state is not None else None

    def save_shard(self, directory, *, rank, world_size, step):
        if self.last_step != int(step):
            raise ValueError(f"EMA/student checkpoint step mismatch: EMA {self.last_step}, student {step}")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"teacher_world_size_{world_size}_rank_{rank}.pt"
        temporary = target.with_suffix(f".pt.tmp.{os.getpid()}")
        torch.save(self.state_dict(rank=rank, world_size=world_size), temporary)
        os.replace(temporary, target)

    def load_shard(self, directory, *, rank, world_size):
        directory = Path(directory)
        manifest = json.loads((directory / "complete.json").read_text())
        if manifest.get("world_size") != int(world_size):
            raise ValueError("EMA checkpoints require the original FSDP world size")
        filename = f"teacher_world_size_{world_size}_rank_{rank}.pt"
        if (directory / filename).stat().st_size != manifest["files"].get(filename):
            raise ValueError("EMA checkpoint shard size differs from its completion manifest")
        state = torch.load(directory / filename, map_location="cpu", weights_only=True)
        self.load_state_dict(state, rank=rank, world_size=world_size, expected_step=manifest["step"])

    def write_complete(self, directory, *, world_size, step):
        """Call on rank zero only, after every rank has passed a save barrier."""
        directory = Path(directory)
        files = [directory / f"teacher_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
        if not all(path.is_file() and path.stat().st_size > 0 for path in files):
            raise RuntimeError("Cannot mark an incomplete EMA checkpoint as complete")
        manifest = dict(format_version=self.format_version, world_size=int(world_size), step=int(step),
                        schedule=self.schedule, beta=self.last_beta,
                        files={path.name: path.stat().st_size for path in files})
        temporary = directory / "complete.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, directory / "complete.json")
