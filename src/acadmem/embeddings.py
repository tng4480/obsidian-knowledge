"""Local embeddings (sentence-transformers). No hosted API is ever called.

``Embedder`` is a small protocol so the indexer/search code does not depend on the concrete
model; tests inject a deterministic fake, production uses :class:`SentenceTransformerEmbedder`
(default ``BAAI/bge-small-en-v1.5``, configurable via ``Config.embedding_model``).

bge models are asymmetric: *queries* get an instruction prefix, *documents* do not. The same
tokenizer used for embedding is exposed through ``count_tokens`` so the chunker's budget matches
what the model will actually see.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import Config


@runtime_checkable
class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str | None = None,
        document_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""
        self._model = None

    # lazily loaded: importing torch/sentence-transformers is slow
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.document_prefix + t for t in texts] if self.document_prefix else texts
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.astype("float32").tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.astype("float32").tolist()

    def count_tokens(self, text: str) -> int:
        tok = self.model.tokenizer
        return len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])


def get_embedder(config: Config) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
        query_prefix=config.query_prefix,
        document_prefix=config.document_prefix,
    )
