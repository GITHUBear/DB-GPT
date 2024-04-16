"""Module for embedding related classes and functions."""

from .embedding_factory import DefaultEmbeddingFactory, EmbeddingFactory  # noqa: F401
from .embeddings import (  # noqa: F401
    Embeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceBGEM3Embeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
    JinaEmbeddings,
    OpenAPIEmbeddings,
)

__ALL__ = [
    "Embeddings",
    "HuggingFaceBgeEmbeddings",
    "HuggingFaceBGEM3Embeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceInferenceAPIEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "JinaEmbeddings",
    "OpenAPIEmbeddings",
    "DefaultEmbeddingFactory",
    "EmbeddingFactory",
]
