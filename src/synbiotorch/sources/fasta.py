"""FASTA corpus source.

One ``Design`` per record. Labels are parsed from ``key=value`` tokens in the
header when ``label_key`` is set. The alphabet is auto-detected (DNA/RNA/protein)
unless overridden, so protein and RNA FASTA feed the right tokenizers.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator, TextIO

from synbiotorch.config import CorpusConfig
from synbiotorch.types import Alphabet, Design, Sequence

from .files import fingerprint_files, list_files

SBOL3 = "http://sbols.org/v3#"
_FASTA_EXTENSIONS = {".fa", ".fasta", ".fna", ".faa"}
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
}
# Characters that, by themselves, identify a sequence as nucleotide (IUPAC).
_NUCLEOTIDE = set("ACGTUNRYSWKMBDHVacgtunryswkmbdhv")


def detect_alphabet(elements: str) -> Alphabet:
    """Infer DNA/RNA/protein from sequence content."""
    letters = {c for c in elements if c.isalpha()}
    if letters and letters <= _NUCLEOTIDE:
        return Alphabet.RNA if ("U" in letters or "u" in letters) else Alphabet.DNA
    return Alphabet.PROTEIN if letters else Alphabet.DNA


class FastaCorpus:
    """Reads ``Design`` records from a FASTA file or a directory of them."""

    def __init__(self, path: str | Path, *, label_key: str | None = None, alphabet: str = "auto") -> None:
        self.path = Path(path)
        self.label_key = label_key
        self.alphabet = alphabet

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "FastaCorpus":
        assert config.path is not None  # guaranteed by CorpusConfig validation
        return cls(config.path, label_key=config.label_key, alphabet=config.alphabet)

    def _files(self) -> list[Path]:
        return [p for p in list_files(self.path) if _is_fasta(p)]

    def _alphabet_for(self, elements: str) -> Alphabet:
        if self.alphabet != "auto":
            return Alphabet(self.alphabet.upper())
        return detect_alphabet(elements)

    def __iter__(self) -> Iterator[Design]:
        for file in self._files():
            yield from self._parse(file)

    def _parse(self, file: Path) -> Iterator[Design]:
        header: str | None = None
        chunks: list[str] = []
        split = _split_from_path(self.path, file)

        def flush() -> Design | None:
            if header is None:
                return None
            seq_id = header.split()[0] if header.split() else header
            label = _label_from_header(header, self.label_key) if self.label_key else None
            elements = "".join(chunks)
            return Design(
                iri=seq_id,
                record_class=f"{SBOL3}Sequence",
                display_id=seq_id,
                sequence=Sequence(elements=elements, alphabet=self._alphabet_for(elements)),
                label=label,
                raw={
                    "header": header,
                    "source_file": str(file),
                    **({"split": split} if split is not None else {}),
                },
            )

        with _open_fasta(file) as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    obj = flush()
                    if obj is not None:
                        yield obj
                    header = line[1:].strip()
                    chunks = []
                elif line:
                    chunks.append(line.strip())
        obj = flush()
        if obj is not None:
            yield obj

    def fingerprint(self) -> str:
        return fingerprint_files(self._files(), self.label_key, self.alphabet)


def _is_fasta(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    return bool(suffixes and suffixes[-1] in _FASTA_EXTENSIONS)


def _open_fasta(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _split_from_path(root: Path, file: Path) -> str | None:
    if not root.is_dir():
        return None
    relative = file.relative_to(root)
    if len(relative.parts) < 2:
        return None
    return _SPLIT_ALIASES.get(relative.parts[0].lower())


def _label_from_header(header: str, label_key: str) -> float | int | None:
    for token in header.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key == label_key:
                try:
                    num = float(value)
                except ValueError:
                    return None
                return int(num) if num.is_integer() else num
    return None
