"""
Integration tests for the encoder-decoder architecture of OmniNet transformer.
"""

import pytest
import torch
import torch.nn as nn

from src.core.encoder_decoder import (
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    TransformerModel
)


@pytest.fixture
def model_config():
    return {
        'num_layers': 2,
        'hidden_size': 256,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'attention_dropout': 0.1,
        'hidden_dropout': 0.1
    }


@pytest.fixture
def sample_batch():
    batch_size = 4
    seq_length = 16
    hidden_size = 256
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'decoder_input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'decoder_attention_mask': torch.ones(batch_size, seq_length)
    }


class TestEncoderLayer:
    """Test suite for EncoderLayer implementation."""

    def test_encoder_layer_forward(self, model_config):
        layer = EncoderLayer(**model_config)
        hidden_states = torch.randn(4, 16, model_config['hidden_size'])
        attention_mask = torch.ones(4, 16)

        output, attention_probs = layer(hidden_states, attention_mask)
        assert output.shape == hidden_states.shape
        assert attention_probs.shape == (4, model_config['num_attention_heads'], 16, 16)

        # Test with attention mask
        attention_mask[:, 8:] = 0
        masked_output, masked_attention_probs = layer(hidden_states, attention_mask)
        assert not torch.allclose(output, masked_output)
        # Verify masked positions have zero attention
        assert torch.all(masked_attention_probs[:, :, :, 8:] < 1e-6)

    def test_residual_connections(self, model_config):
        layer = EncoderLayer(**model_config)
        hidden_states = torch.randn(4, 16, model_config['hidden_size'])
        attention_mask = torch.ones(4, 16)

        # Store intermediate values for Pre-LN architecture verification
        layer.attention.train()  # Enable dropout
        normed_hidden = layer.attention_norm(hidden_states)
        attention_output = layer.attention(normed_hidden, attention_mask)[0]
        attention_residual = hidden_states + layer.dropout(attention_output)

        # Verify Pre-LN residual connections
        final_output, _ = layer(hidden_states, attention_mask)
        normed_attention = layer.feedforward_norm(attention_residual)
        feedforward_output = layer.feedforward(normed_attention)
        expected_output = attention_residual + layer.dropout(feedforward_output)

        assert torch.allclose(final_output, expected_output, rtol=1e-5)


class TestDecoderLayer:
    """Test suite for DecoderLayer implementation."""

    def test_decoder_layer_forward(self, model_config):
        layer = DecoderLayer(**model_config)
        hidden_states = torch.randn(4, 16, model_config['hidden_size'])
        encoder_hidden_states = torch.randn(4, 16, model_config['hidden_size'])
        attention_mask = torch.ones(4, 16)
        encoder_attention_mask = torch.ones(4, 16)

        output, self_attn, cross_attn = layer(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask
        )
        assert output.shape == hidden_states.shape

    def test_causal_attention(self, model_config):
        layer = DecoderLayer(**model_config)
        batch_size = 4
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, model_config['hidden_size'])
        encoder_hidden_states = torch.randn(batch_size, seq_length, model_config['hidden_size'])

        # Create causal attention mask with proper broadcasting dimensions
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).expand(batch_size, -1, -1)

        output_with_causal, _, _ = layer(
            hidden_states,
            encoder_hidden_states,
            causal_mask,
            None
        )
        output_without_causal, _, _ = layer(
            hidden_states,
            encoder_hidden_states,
            None,
            None
        )

        assert not torch.allclose(output_with_causal, output_without_causal)


class TestTransformerModel:
    """Test suite for complete TransformerModel."""

    def test_model_forward(self, model_config, sample_batch):
        model = TransformerModel(**model_config)
        outputs = model(**sample_batch)

        # Unpack outputs - model returns (decoder_output, encoder_attn, decoder_self_attn, decoder_cross_attn)
        decoder_output = outputs[0]
        assert decoder_output.shape == (
            sample_batch['input_ids'].size(0),
            sample_batch['input_ids'].size(1),
            model_config['hidden_size']
        )

    def test_attention_head_pruning(self, model_config, sample_batch):
        model = TransformerModel(**model_config, head_pruning=True)
        
        # Get initial head weights
        encoder_head_weights = []
        for layer in model.encoder.layers:
            encoder_head_weights.append(layer.attention.head_weights.clone())
        
        # Forward pass should update head weights
        _ = model(**sample_batch)
        
        # Check if head weights changed
        for i, layer in enumerate(model.encoder.layers):
            assert not torch.allclose(
                encoder_head_weights[i],
                layer.attention.head_weights
            )

    def test_extended_sequence_length(self, model_config):
        model = TransformerModel(**model_config)
        
        # Test with longer sequence
        batch_size = 2
        long_seq_length = model_config['max_position_embeddings'] - 100
        long_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, long_seq_length)),
            'attention_mask': torch.ones(batch_size, long_seq_length),
            'decoder_input_ids': torch.randint(0, 1000, (batch_size, long_seq_length)),
            'decoder_attention_mask': torch.ones(batch_size, long_seq_length)
        }
        
        outputs = model(**long_input)
        assert outputs.shape == (batch_size, long_seq_length, model_config['hidden_size'])

    def test_memory_efficient_attention(self, model_config, sample_batch):
        # Test with memory efficient attention
        model = TransformerModel(
            **model_config,
            use_memory_efficient_attention=True
        )
        
        # Record memory usage before forward pass
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        outputs = model(**sample_batch)
        
        # Verify outputs and memory usage
        assert outputs.shape == (
            sample_batch['input_ids'].size(0),
            sample_batch['input_ids'].size(1),
            model_config['hidden_size']
        )
        current_memory = torch.cuda.memory_allocated()
        assert current_memory - initial_memory < 1e6  # Less than 1MB increase


if __name__ == "__main__":
    pytest.main([__file__])
