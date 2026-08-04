"""Overlapping k-mer tokenizer for nucleotide sequences.

Every k-mer over the {A,C,G,T} alphabet is a vocabulary entry. Ambiguous IUPAC
bases (N, R, Y, ...) fall back to the ``<unk>`` token rather than being treated
as a real base, which SeqTrainer's one-hot path got wrong.
"""

from __future__ import annotations

from itertools import product

from synbiotorch.tokenize.base import Encoded, SpannedTokens

_BASES = "ACGT"
# Reserved ids: 0 pad, 1 unk, 2 cls, 3 sep, 4 mask. Real k-mers follow.
_SPECIAL = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]


class KmerTokenizer:
    def __init__(self, k: int = 6, stride: int = 1, max_length: int = 512) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.stride = stride
        self._max_length = max_length
        kmers = ["".join(p) for p in product(_BASES, repeat=k)]
        self._vocab = {tok: i for i, tok in enumerate(_SPECIAL + kmers)}
        self._id_to_tok = {i: tok for tok, i in self._vocab.items()}
        self._unk = self._vocab["<unk>"]
        self._special_ids = frozenset(self._vocab[tok] for tok in _SPECIAL)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return self._vocab["<pad>"]

    @property
    def mask_token_id(self) -> int | None:
        return self._vocab["<mask>"]

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def max_length(self) -> int:
        return self._max_length

    def tokenize_content(self, sequence: str) -> list[int]:
        seq = sequence.upper()
        return [
            self._vocab.get(seq[start : start + self.k], self._unk)
            for start in range(0, max(0, len(seq) - self.k + 1), self.stride)
        ]

    def encode(self, sequence: str) -> Encoded:
        content = self.tokenize_content(sequence)[: self._max_length - 2]
        ids = [self._vocab["<cls>"], *content, self._vocab["<sep>"]]
        return Encoded(input_ids=ids, attention_mask=[1] * len(ids))

    def decode(self, ids: list[int]) -> str:
        """Reconstruct the sequence from k-mer ids.

        For the default stride of 1, adjacent k-mers overlap by ``k-1`` bases, so
        the sequence is the first k-mer plus the last base of each subsequent one;
        ``<unk>`` and special tokens are dropped. With a larger stride the overlap
        is lost, so k-mers are concatenated (lossy).
        """
        toks = [
            self._id_to_tok[i] for i in ids if i not in self._special_ids and self._id_to_tok.get(i, "<unk>") != "<unk>"
        ]
        if not toks:
            return ""
        if self.stride == 1:
            return toks[0] + "".join(tok[-1] for tok in toks[1:])
        return "".join(toks)


class NonOverlappingKmerTokenizer:
    """Lossless fixed-vocabulary non-overlapping k-mers for genomic streams.

    Canonical full-width chunks become one k-mer token. Ambiguous bases and a
    trailing partial chunk fall back one base at a time, so encoding never drops
    source sequence. For ``k=6`` the vocabulary contains pad, unknown,
    N/A/C/G/T, then all 4,096 canonical six-mers.
    """

    _SINGLE = {"N": 2, "A": 3, "C": 4, "G": 5, "T": 6}
    _ID_TO_SINGLE = {value: key for key, value in _SINGLE.items()}
    _OFFSET = 7

    def __init__(self, k: int = 6, max_length: int = 512) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.k = k
        self._max_length = max_length
        self._canonical_count = 4**k
        # Single-base fallbacks are biological content, not control tokens.
        self._special_ids = frozenset({0, 1})

    @property
    def vocab_size(self) -> int:
        return self._OFFSET + self._canonical_count

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def mask_token_id(self) -> int | None:
        return None

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def max_length(self) -> int:
        return self._max_length

    @staticmethod
    def _normalize(sequence: str) -> str:
        return "".join(base if base in "ACGTN" else "N" for base in sequence.upper())

    def _kmer_id(self, kmer: str) -> int:
        value = 0
        for base in kmer:
            value = value * 4 + _BASES.index(base)
        return self._OFFSET + value

    def _decode_kmer(self, token_id: int) -> str:
        value = token_id - self._OFFSET
        bases = ["A"] * self.k
        for index in range(self.k - 1, -1, -1):
            bases[index] = _BASES[value % 4]
            value //= 4
        return "".join(bases)

    def tokenize_content_with_spans(self, sequence: str) -> SpannedTokens:
        normalized = self._normalize(sequence)
        ids: list[int] = []
        counts: list[int] = []
        position = 0
        while position < len(normalized):
            chunk = normalized[position : position + self.k]
            if len(chunk) == self.k and set(chunk).issubset(_BASES):
                ids.append(self._kmer_id(chunk))
                counts.append(self.k)
                position += self.k
            else:
                ids.append(self._SINGLE[normalized[position]])
                counts.append(1)
                position += 1
        return SpannedTokens(ids, counts)

    def tokenize_content(self, sequence: str) -> list[int]:
        return self.tokenize_content_with_spans(sequence).input_ids

    def encode(self, sequence: str) -> Encoded:
        content = self.tokenize_content(sequence)[: self._max_length]
        return Encoded(input_ids=content, attention_mask=[1] * len(content))

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                continue
            if token_id in self._ID_TO_SINGLE:
                pieces.append(self._ID_TO_SINGLE[token_id])
            elif self._OFFSET <= token_id < self.vocab_size:
                pieces.append(self._decode_kmer(token_id))
            else:
                pieces.append("N")
        return "".join(pieces)
