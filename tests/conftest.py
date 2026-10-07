"""Shared fixtures: synthetic PDFs, a deterministic fake embedder, a temp vault."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_pdf(
    path: Path,
    pages: list[dict],
    header: str | None = "COURSE 3B1 / PAPER 2023",
    footer: str | None = "WORKED GUIDE",
    title: str = "Test Guide",
) -> Path:
    """Create a PDF. Each page dict: ``{"heading": str|None, "outline": str|None, "body": [str, ...]}``.

    A page with an ``outline`` value gets a level-1 PDF bookmark. Header/footer text and a page number
    repeat on every page (to exercise boilerplate stripping).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(title)
    width, height = A4
    for i, page in enumerate(pages, start=1):
        if header:
            c.setFont("Helvetica", 8)
            c.drawString(50, height - 30, header)
        y = height - 90
        if page.get("outline"):
            key = f"bm{i}"
            c.bookmarkPage(key)
            c.addOutlineEntry(page["outline"], key, level=0)
        if page.get("heading"):
            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, y, page["heading"])
            y -= 40
        c.setFont("Helvetica", 11)
        for line in page.get("body", []):
            c.drawString(50, y, line)
            y -= 18
        if footer:
            c.setFont("Helvetica", 8)
            c.drawString(50, 30, f"{footer}   {i:02d}")
        c.showPage()
    c.save()
    return path


@pytest.fixture
def pdf_factory() -> Callable[..., Path]:
    return make_pdf


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download).

    Texts that share vocabulary have high cosine similarity, which is enough to exercise the
    indexing / retrieval / filtering / expansion plumbing quickly. Real-model behaviour is covered
    by ``@pytest.mark.slow`` tests.
    """

    dim = 256

    def __init__(self) -> None:
        self.model_name = "fake-hash-embedder"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def count_tokens(self, text: str) -> int:
        return len(self._tokens(text))

    @property
    def dimension(self) -> int:
        return self.dim


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


BAYES_MD = """---
tags: [probability]
---
# Bayesian inference

Bayes theorem updates a prior belief with the likelihood of observed data to give the posterior.

## What I don't understand

Why does the evidence term normalise the posterior distribution? I struggled with the denominator.

## Worked example

A medical test has a false positive rate. The posterior probability of disease given a positive test is small
because the prior prevalence of the disease is low.
"""

MARKOV_MD = """# Markov chains

A Markov chain has the memoryless property: the next state depends only on the current state.

## Stationary distribution

The stationary distribution satisfies pi P = pi and describes the long run fraction of time in each state.
"""

PDF_PAGES = [
    {
        "outline": "Conditional probability",
        "heading": "Conditional probability",
        "body": [
            "Conditional probability measures the chance of an event given another event has occurred.",
            "P(A|B) = P(A and B) / P(B) whenever P(B) is positive.",
        ],
    },
    {"body": ["The law of total probability partitions the sample space into disjoint events."]},
    {
        "outline": "Independence",
        "heading": "Independence",
        "body": ["Two events are independent when the occurrence of one does not change the probability of the other."],
    },
    {"body": ["Pairwise independence is weaker than mutual independence of a whole family of events."]},
]


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A small multi-module vault: Markdown + PDF, nested folders, ignored dirs, an empty file."""
    root = tmp_path / "vault"
    (root / ".obsidian").mkdir(parents=True)
    (root / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    (root / "3B1" / "Questions").mkdir(parents=True)
    (root / "3B1" / "Notes").mkdir(parents=True)
    (root / "3F1" / "Notes" / "deep" / "er").mkdir(parents=True)
    (root / "3B1" / "Questions" / "Bayes.md").write_text(BAYES_MD, encoding="utf-8")
    (root / "3F1" / "Notes" / "deep" / "er" / "Markov.md").write_text(MARKOV_MD, encoding="utf-8")
    (root / "3B1" / "Notes" / "_folder.md").write_text("", encoding="utf-8")  # empty placeholder
    (root / "loose.md").write_text("# Loose\n\nA note at the vault root about queueing theory.\n", encoding="utf-8")
    make_pdf(root / "3B1" / "Handouts" / "Week 5 - Probability.pdf", PDF_PAGES)
    return root


@pytest.fixture
def config(vault: Path, tmp_path: Path):
    from acadmem.config import Config

    return Config(
        vault_path=vault,
        index_dir=tmp_path / "index",
        max_chunk_tokens=120,
        min_chunk_tokens=10,
        overlap_tokens=10,
        watch_debounce_seconds=0.2,
    )


@pytest.fixture
def indexer(config, fake_embedder):
    from acadmem.indexer import Indexer

    return Indexer(config, embedder=fake_embedder)
