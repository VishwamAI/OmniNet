"""
Unit tests for the base layers of the OmniNet transformer architecture.
"""

import pytest
import torch
import torch.nn as nn

from src.core.base_layers import (
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding
)


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_length():
    return 32


@pytest.fixture
def hidden_size():
    return 256


@pytest.fixture
def num_attention_heads():
    return 8


@pytest.fixture
def sample_hidden_states(batch_size, seq_length, hidden_size):
    return torch.randn(batch_size, seq_length, hidden_size)


class TestLayerNorm:
    """Test suite for LayerNorm implementation."""

    def test_layer_norm_shape(self, sample_hidden_states):
        layer_norm = LayerNorm(sample_hidden_states.size(-1))
        output = layer_norm(sample_hidden_states)
        assert output.shape == sample_hidden_states.shape

    def test_layer_norm_mean_var(self, sample_hidden_states):
        layer_norm = LayerNorm(sample_hidden_states.size(-1))
        output = layer_norm(sample_hidden_states)
        # Check if normalized (mean ≈ 0, var ≈ 1)
        mean = output.mean(dim=-1)
        var = output.var(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention implementation."""

    @pytest.fixture
    def attention_module(self, hidden_size, num_attention_heads):
        return MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads
        )

    def test_attention_shape(self, attention_module, sample_hidden_states):
        output, attention_probs = attention_module(sample_hidden_states)
        assert output.shape == sample_hidden_states.shape
        assert attention_probs.shape == (
            sample_hidden_states.size(0),
            attention_module.num_attention_heads,
            sample_hidden_states.size(1),
            sample_hidden_states.size(1)
        )

    def test_attention_mask(self, attention_module, sample_hidden_states):
        batch_size, seq_length = sample_hidden_states.shape[:2]
        attention_mask = torch.zeros(batch_size, seq_length)
        attention_mask[:, :seq_length//2] = float('-inf')
        
        output, attention_probs = attention_module(
            sample_hidden_states,
            attention_mask=attention_mask
        )
        # Check if masked positions have zero attention
        assert torch.all(attention_probs[:, :, :, :seq_length//2] < 1e-6)

    def test_head_pruning(self, hidden_size, num_attention_heads):
        attention_module = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_pruning=True
        )
        assert hasattr(attention_module, 'head_weights')
        assert attention_module.head_weights.shape == (num_attention_heads,)


class TestFeedForward:
    """Test suite for FeedForward implementation."""

    @pytest.fixture
    def ff_module(self, hidden_size):
        return FeedForward(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4
        )

    def test_feedforward_shape(self, ff_module, sample_hidden_states):
        output = ff_module(sample_hidden_states)
        assert output.shape == sample_hidden_states.shape

    def test_activation_functions(self, hidden_size):
        # Test different activation functions
        ff_relu = FeedForward(hidden_size, hidden_size * 4, activation="relu")
        ff_gelu = FeedForward(hidden_size, hidden_size * 4, activation="gelu")
        
        input_tensor = torch.randn(2, 4, hidden_size)
        assert ff_relu(input_tensor).shape == input_tensor.shape
        assert ff_gelu(input_tensor).shape == input_tensor.shape

    def test_dropout(self, ff_module, sample_hidden_states):
        ff_module.train()  # Set to training mode
        output1 = ff_module(sample_hidden_states)
        output2 = ff_module(sample_hidden_states)
        # Outputs should be different during training due to dropout
        assert not torch.allclose(output1, output2)


class TestPositionalEncoding:
    """Test suite for PositionalEncoding implementation."""

    @pytest.fixture
    def pos_encoding(self, hidden_size):
        return PositionalEncoding(
            hidden_size=hidden_size,
            max_position_embeddings=512
        )

    def test_positional_encoding_shape(self, pos_encoding, sample_hidden_states):
        output = pos_encoding(sample_hidden_states)
        assert output.shape == sample_hidden_states.shape

    def test_learnable_parameters(self, pos_encoding):
        # Check if position embeddings are learnable
        assert isinstance(pos_encoding.position_embeddings, nn.Parameter)
        assert pos_encoding.position_embeddings.requires_grad

    def test_max_sequence_length(self, hidden_size):
        max_length = 1024
        pos_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_position_embeddings=max_length
        )
        # Test with sequence longer than usual but shorter than max
        long_sequence = torch.randn(2, max_length - 100, hidden_size)
        output = pos_encoding(long_sequence)
        assert output.shape == long_sequence.shape


if __name__ == "__main__":
    pytest.main([__file__])
