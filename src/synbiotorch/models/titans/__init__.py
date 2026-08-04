"""Paper-deep TITANS Memory-as-Context architecture."""

from synbiotorch.models.titans.causal import TitansCausalLM, TitansCausalOutput
from synbiotorch.models.titans.mac import MACBlock, MACOutput, mac_attention_mask
from synbiotorch.models.titans.memory import NeuralMemory, PaperResidualMemory
from synbiotorch.models.titans.state import NeuralMemoryState, TitansState

__all__ = [
    "TitansCausalLM",
    "TitansCausalOutput",
    "TitansState",
    "NeuralMemoryState",
    "NeuralMemory",
    "PaperResidualMemory",
    "MACBlock",
    "MACOutput",
    "mac_attention_mask",
]
