"""Exact paper-deep neural memory used by the TITANS MAC block."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, cast

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from synbiotorch.models.titans.state import NeuralMemoryState, TensorMap


class PaperResidualMemory(nn.Module):
    """Two-linear-layer, expanded residual MLP used as functional fast memory."""

    def __init__(self, d_model: int, expansion_factor: int) -> None:
        super().__init__()
        hidden_size = d_model * expansion_factor
        self.in_projection = nn.Linear(d_model, hidden_size)
        self.out_projection = nn.Linear(hidden_size, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.out_projection(F.gelu(self.in_projection(inputs)))
        return inputs + self.normalization(transformed)


@dataclass(frozen=True)
class ParameterGates:
    alpha: Mapping[str, Tensor]
    eta: Mapping[str, Tensor]
    theta: Mapping[str, Tensor]


class ChannelUpdateGates(nn.Module):
    """Input-dependent alpha/eta/theta per memory layer output channel."""

    _LAYERS = {
        "in_projection.weight": "in_projection",
        "in_projection.bias": "in_projection",
        "out_projection.weight": "out_projection",
        "out_projection.bias": "out_projection",
        "normalization.weight": "out_projection",
        "normalization.bias": "out_projection",
    }

    def __init__(
        self,
        d_model: int,
        expansion_factor: int,
        *,
        alpha_initial: float,
        eta_initial: float,
        theta_initial: float,
    ) -> None:
        super().__init__()
        widths = {
            "in_projection": d_model * expansion_factor,
            "out_projection": d_model,
        }
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, 3 * width) for name, width in widths.items()})
        initials = (alpha_initial, eta_initial, theta_initial)
        with torch.no_grad():
            for module in self.heads.values():
                head = cast(nn.Linear, module)
                width = head.out_features // 3
                head.weight.zero_()
                for index, value in enumerate(initials):
                    start = index * width
                    head.bias[start : start + width].fill_(torch.logit(torch.tensor(value)).item())

    def forward(self, inputs: Tensor) -> ParameterGates:
        per_layer: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for name, module in self.heads.items():
            head = cast(nn.Linear, module)
            alpha, eta, theta = torch.sigmoid(head(inputs)).chunk(3, dim=-1)
            per_layer[name] = (alpha, eta, theta)
        result: dict[str, OrderedDict[str, Tensor]] = {name: OrderedDict() for name in ("alpha", "eta", "theta")}
        for parameter_name, layer_name in self._LAYERS.items():
            is_matrix = parameter_name.endswith(".weight") and not parameter_name.startswith("normalization")
            for gate_index, gate_name in enumerate(("alpha", "eta", "theta")):
                value = per_layer[layer_name][gate_index]
                result[gate_name][parameter_name] = value.unsqueeze(-1) if is_matrix else value
        return ParameterGates(**result)


class NeuralMemory(nn.Module):
    """Stream-local functional MLP with exact sequential surprise updates."""

    def __init__(
        self,
        d_model: int,
        *,
        expansion_factor: int = 4,
        projection_kernel_size: int = 4,
        alpha_initial: float = 0.001,
        eta_initial: float = 0.9,
        theta_initial: float = 0.001,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.projection_kernel_size = projection_kernel_size
        self.memory = PaperResidualMemory(d_model, expansion_factor)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        def convolution() -> nn.Conv1d:
            layer = nn.Conv1d(
                d_model,
                d_model,
                projection_kernel_size,
                groups=d_model,
                bias=False,
            )
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight[:, 0, -1] = 1.0
            return layer

        self.query_convolution = convolution()
        self.key_convolution = convolution()
        self.value_convolution = convolution()
        self.gates = ChannelUpdateGates(
            d_model,
            expansion_factor,
            alpha_initial=alpha_initial,
            eta_initial=eta_initial,
            theta_initial=theta_initial,
        )

    def initial_state(self) -> NeuralMemoryState:
        fast_weights = OrderedDict(self.memory.named_parameters())
        reference = next(iter(fast_weights.values()))
        history = reference.new_zeros((self.projection_kernel_size - 1, self.d_model))
        return NeuralMemoryState(
            fast_weights=fast_weights,
            surprise=OrderedDict((name, torch.zeros_like(value)) for name, value in fast_weights.items()),
            query_history=history,
            write_history=history.clone(),
        )

    def _functional_memory(self, weights: Mapping[str, Tensor], inputs: Tensor) -> Tensor:
        return functional_call(self.memory, OrderedDict(weights.items()), (inputs,), strict=True)

    def _memory_inputs(self, inputs: Tensor) -> Tensor:
        """Keep fast weights and their higher-order update arithmetic in one dtype."""

        return inputs.to(dtype=next(self.memory.parameters()).dtype)

    def _project(
        self,
        projection: nn.Linear,
        convolution: nn.Conv1d,
        inputs: Tensor,
        history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = self.projection_kernel_size - 1
        if history.shape != (expected, self.d_model):
            raise ValueError("causal projection history has the wrong shape")
        combined = torch.cat((history.to(inputs), inputs), dim=0)
        projected = projection(combined).transpose(0, 1).unsqueeze(0)
        convolved = convolution(projected).squeeze(0).transpose(0, 1)
        return convolved, combined[-expected:]

    @staticmethod
    def _unit(values: Tensor) -> Tensor:
        epsilon = torch.finfo(values.dtype).eps
        return values / values.norm(dim=-1, keepdim=True).clamp_min(epsilon)

    def read_with_history(self, state: NeuralMemoryState, inputs: Tensor) -> tuple[Tensor, Tensor]:
        # Attention can use the enclosing AMP policy, but the functional fast
        # weights and autograd-through-update recurrence remain parameter-dtype.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            memory_inputs = self._memory_inputs(inputs)
            queries, history = self._project(
                self.query_projection,
                self.query_convolution,
                memory_inputs,
                state.query_history,
            )
            return self._functional_memory(state.fast_weights, self._unit(queries)), history

    def _project_writes(self, state: NeuralMemoryState, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        keys, history = self._project(
            self.key_projection,
            self.key_convolution,
            inputs,
            state.write_history,
        )
        values, value_history = self._project(
            self.value_projection,
            self.value_convolution,
            inputs,
            state.write_history,
        )
        if not torch.equal(history, value_history):
            raise RuntimeError("key and value convolution histories diverged")
        return self._unit(keys), values, history

    def associative_loss(self, fast_weights: Mapping[str, Tensor], key: Tensor, value: Tensor) -> Tensor:
        reconstruction = self._functional_memory(fast_weights, key)
        return 0.5 * (reconstruction - value).square().sum()

    def _surprise_gradient(
        self,
        fast_weights: Mapping[str, Tensor],
        key: Tensor,
        value: Tensor,
        *,
        create_graph: bool,
    ) -> TensorMap:
        with torch.enable_grad():
            loss = self.associative_loss(fast_weights, key, value)
            gradients = torch.autograd.grad(
                loss,
                tuple(fast_weights.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )
        return OrderedDict(zip(fast_weights, gradients))

    def update(
        self,
        state: NeuralMemoryState,
        inputs: Tensor,
        *,
        valid_mask: Tensor,
        query_history: Tensor,
        create_graph: bool,
    ) -> NeuralMemoryState:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            memory_inputs = self._memory_inputs(inputs)
            keys, values, write_history = self._project_writes(state, memory_inputs)
            gates = self.gates(memory_inputs)
            fast_weights = OrderedDict(state.fast_weights.items())
            surprise = OrderedDict(state.surprise.items())
            for position in range(memory_inputs.shape[0]):
                if not bool(valid_mask[position].item()):
                    continue
                gradient = self._surprise_gradient(
                    fast_weights,
                    keys[position],
                    values[position],
                    create_graph=create_graph,
                )
                next_surprise: TensorMap = OrderedDict()
                next_weights: TensorMap = OrderedDict()
                for name in fast_weights:
                    alpha = gates.alpha[name][position]
                    eta = gates.eta[name][position]
                    theta = gates.theta[name][position]
                    next_surprise[name] = eta * surprise[name] - theta * gradient[name]
                    next_weights[name] = (1.0 - alpha) * fast_weights[name] + next_surprise[name]
                if not create_graph:
                    next_weights = OrderedDict(
                        (name, value.detach().requires_grad_(True)) for name, value in next_weights.items()
                    )
                    next_surprise = OrderedDict((name, value.detach()) for name, value in next_surprise.items())
                fast_weights, surprise = next_weights, next_surprise
            return NeuralMemoryState(
                fast_weights=fast_weights,
                surprise=surprise,
                query_history=query_history,
                write_history=write_history,
            )

    def freeze(
        self,
        state: NeuralMemoryState,
        inputs: Tensor,
        *,
        query_history: Tensor,
    ) -> NeuralMemoryState:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            _, _, write_history = self._project_writes(state, self._memory_inputs(inputs))
            return NeuralMemoryState(
                fast_weights=state.fast_weights,
                surprise=state.surprise,
                query_history=query_history,
                write_history=write_history,
            )
