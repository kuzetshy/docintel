"""Local embeddings with fastembed (ONNX on CPU, no API key, no cost).

This is the only module allowed to import `fastembed`.
"""

from collections.abc import Sequence

from fastembed import TextEmbedding


class FastEmbedEmbedder:
    """Embedder backed by a fastembed model.

    The model (~70 MB for bge-small) is downloaded on first use and cached,
    so creating the object is cheap and the first `embed_*` call is slow.
    """

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._dimension = TextEmbedding.get_embedding_size(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().passage_embed(list(texts))
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._get_model().query_embed(text)))
        return vector.tolist()  # type: ignore[no-any-return]
