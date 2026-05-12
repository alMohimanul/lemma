"""Tests for embeddings module."""
import numpy as np
from MAVYN.embeddings.encoder import EmbeddingEncoder


def test_encoder_initialization():
    """Test encoder can be initialized."""
    encoder = EmbeddingEncoder()
    assert encoder is not None
    assert encoder.model is not None


def test_encode_text():
    """Test encoding text to embeddings."""
    encoder = EmbeddingEncoder()
    text = "This is a test sentence for embedding."

    embedding = encoder.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0
    assert embedding.dtype == np.float32


def test_encode_batch():
    """Test batch encoding of multiple texts."""
    encoder = EmbeddingEncoder()
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]

    embeddings = encoder.encode_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0


def test_embedding_consistency():
    """Test that same text produces same embedding."""
    encoder = EmbeddingEncoder()
    text = "Consistency test sentence."

    embedding1 = encoder.encode(text)
    embedding2 = encoder.encode(text)

    # Should be very similar (allowing for floating point precision)
    assert np.allclose(embedding1, embedding2, rtol=1e-5)


def test_encode_empty_text():
    """Test encoding empty text."""
    encoder = EmbeddingEncoder()

    # Empty string should still produce an embedding
    embedding = encoder.encode("")

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


def test_similarity_calculation():
    """Test that similar texts produce similar embeddings."""
    encoder = EmbeddingEncoder()

    text1 = "Machine learning is fascinating."
    text2 = "I find machine learning very interesting."
    text3 = "The weather is nice today."

    emb1 = encoder.encode(text1)
    emb2 = encoder.encode(text2)
    emb3 = encoder.encode(text3)

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    # Similar texts should have higher similarity
    assert sim_12 > sim_13


def test_embedding_dimensions():
    """Test that embeddings have expected dimensions."""
    encoder = EmbeddingEncoder()
    text = "Test for dimension check."

    embedding = encoder.encode(text)

    assert embedding.shape[0] == encoder.embedding_dim


def test_normalize_embeddings():
    """Test that embeddings can be normalized."""
    encoder = EmbeddingEncoder()
    text = "Normalization test."

    embedding = encoder.encode(text)

    # Normalize
    normalized = embedding / np.linalg.norm(embedding)

    # Check that norm is 1
    assert np.isclose(np.linalg.norm(normalized), 1.0)
