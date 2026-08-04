"""Tokenizers: one protocol, swappable implementations."""

from __future__ import annotations

from synbiotorch.tokenize.base import Encoded, SpannedTokenizer, SpannedTokens, Tokenizer, build_tokenizer
from synbiotorch.tokenize.char import CharTokenizer
from synbiotorch.tokenize.hf import HFTokenizer
from synbiotorch.tokenize.kmer import KmerTokenizer, NonOverlappingKmerTokenizer

__all__ = [
    "Encoded",
    "SpannedTokens",
    "SpannedTokenizer",
    "Tokenizer",
    "build_tokenizer",
    "CharTokenizer",
    "HFTokenizer",
    "KmerTokenizer",
    "NonOverlappingKmerTokenizer",
]
