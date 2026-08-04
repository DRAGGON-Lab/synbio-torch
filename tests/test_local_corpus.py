from __future__ import annotations

import gzip

from synbiotorch.sources.fasta import FastaCorpus
from synbiotorch.sources.sbol import SbolFileCorpus


def test_fasta_parsing_with_labels(fasta_file):
    corpus = FastaCorpus(fasta_file, label_key="measure")
    objects = list(corpus)
    assert len(objects) == 3
    assert objects[0].display_id == "seq1"
    assert objects[0].sequence.elements == "ACGTACGTACGTACGT"
    assert objects[0].label == 12.5
    assert objects[2].label == 8.25


def test_fasta_without_label_key_is_unlabeled(fasta_file):
    corpus = FastaCorpus(fasta_file)
    objects = list(corpus)
    assert all(o.label is None for o in objects)


def test_sbol_parsing(sbol_file):
    corpus = SbolFileCorpus(sbol_file, label_key="measure")
    objects = sorted(corpus, key=lambda o: o.iri)
    assert len(objects) == 2
    assert objects[0].sequence.elements.startswith("ttgacg")
    assert objects[0].label == 5.0


def test_fingerprint_is_stable(fasta_file):
    a = FastaCorpus(fasta_file).fingerprint()
    b = FastaCorpus(fasta_file).fingerprint()
    assert a == b


def test_compressed_fasta_split_directories_preserve_stream_boundaries(tmp_path):
    for split, accession in (("train", "GCF_000005845.2"), ("val", "GCA_000249155.2")):
        directory = tmp_path / split
        directory.mkdir()
        path = directory / f"{accession}_genomic.fna.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(f">{accession}_chromosome complete genome\nACGTACGT\n")
    objects = list(FastaCorpus(tmp_path))
    assert [obj.raw["split"] for obj in objects] == ["train", "val"]
    assert [obj.raw["source_file"] for obj in objects] == [
        str(tmp_path / "train" / "GCF_000005845.2_genomic.fna.gz"),
        str(tmp_path / "val" / "GCA_000249155.2_genomic.fna.gz"),
    ]
