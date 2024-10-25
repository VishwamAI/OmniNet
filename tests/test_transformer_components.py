import pytest
import torch
import torch.nn as nn

from src.core.base_layers import (
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding
)
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
        'batch_size': 4,
        'seq_length': 16,
        'hidden_size': 256,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'attention_dropout': 0.1,
        'hidden_dropout': 0.1
    }

def test_layer_norm(model_config):
    hidden_size = model_config['hidden_size']
    layer_norm = LayerNorm(hidden_size)
    x = torch.randn(model_config['batch_size'], model_config['seq_length'], hidden_size)
    output = layer_norm(x)

    # Test output shape
    assert output.shape == x.shape
    # Test normalization (with adjusted tolerances)
    assert torch.allclose(output.mean(-1), torch.zeros_like(output.mean(-1)), atol=1e-4)
    assert torch.allclose(output.std(-1), torch.ones_like(output.std(-1)), atol=1e-2)

def test_positional_encoding(model_config):
    hidden_size = model_config['hidden_size']
    max_position_embeddings = model_config['max_position_embeddings']
    pos_encoding = PositionalEncoding(hidden_size, max_position_embeddings)
    x = torch.randn(model_config['batch_size'], model_config['seq_length'], hidden_size)
    output = pos_encoding(x)

    # Test output shape
    assert output.shape == x.shape
    # Test learnable embeddings
    assert isinstance(pos_encoding.position_embeddings, nn.Parameter)

def test_multi_head_attention(model_config):
    hidden_size = model_config['hidden_size']
    num_attention_heads = model_config['num_attention_heads']
    attention = MultiHeadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_dropout=model_config['attention_dropout']
    )
    x = torch.randn(model_config['batch_size'], model_config['seq_length'], hidden_size)
    output, attention_probs = attention(x)

    # Test output shape
    assert output.shape == x.shape
    # Test attention probabilities shape
    assert attention_probs.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])

def test_feed_forward(model_config):
    hidden_size = model_config['hidden_size']
    intermediate_size = model_config['intermediate_size']
    feed_forward = FeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_dropout=model_config['hidden_dropout']
    )
    x = torch.randn(model_config['batch_size'], model_config['seq_length'], hidden_size)
    output = feed_forward(x)

    # Test output shape
    assert output.shape == x.shape

def test_encoder_layer(model_config):
    encoder_layer = EncoderLayer(
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
    x = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    output, attention_probs = encoder_layer(x)

    # Test output shape
    assert output.shape == x.shape
    # Test attention probabilities shape
    assert attention_probs.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])

def test_decoder_layer(model_config):
    decoder_layer = DecoderLayer(
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
    x = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    encoder_output = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    output, self_attn_probs, cross_attn_probs = decoder_layer(x, encoder_output)

    # Test output shape
    assert output.shape == x.shape
    # Test attention probabilities shapes
    assert self_attn_probs.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
    assert cross_attn_probs.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])

def test_encoder(model_config):
    encoder = Encoder(
        num_layers=2,
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
    x = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    output, attention_probs = encoder(x)

    # Test output shape
    assert output.shape == x.shape
    # Test attention probabilities shape
    assert all(layer_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_attn in attention_probs)

def test_decoder(model_config):
    decoder = Decoder(
        num_layers=2,
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
    x = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    encoder_output = torch.randn(
        model_config['batch_size'],
        model_config['seq_length'],
        model_config['hidden_size']
    )
    output, self_attn_probs, cross_attn_probs = decoder(x, encoder_output)

    # Test output shape
    assert output.shape == x.shape
    # Test attention probabilities shapes
    assert all(layer_self_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_self_attn in self_attn_probs)
    assert all(layer_cross_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_cross_attn in cross_attn_probs)

def test_transformer_model(model_config):
    model = TransformerModel(
        num_layers=2,
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
        attention_dropout=model_config['attention_dropout'],
        hidden_dropout=model_config['hidden_dropout']
    )
    encoder_input = torch.randint(
        0, 1000,  # Vocabulary range
        (model_config['batch_size'], model_config['seq_length']),
        dtype=torch.long
    )
    decoder_input = torch.randint(
        0, 1000,  # Vocabulary range
        (model_config['batch_size'], model_config['seq_length']),
        dtype=torch.long
    )
    output, encoder_attn_probs, decoder_self_attn_probs, decoder_cross_attn_probs = model(
        input_ids=encoder_input,
        decoder_input_ids=decoder_input
    )

    # Test output shape
    assert output.shape == (model_config['batch_size'], model_config['seq_length'], model_config['hidden_size'])
    # Test attention probabilities shapes
    assert all(layer_enc_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_enc_attn in encoder_attn_probs)
    assert all(layer_dec_self_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_dec_self_attn in decoder_self_attn_probs)
    assert all(layer_dec_cross_attn.shape == (model_config['batch_size'], model_config['num_attention_heads'], model_config['seq_length'], model_config['seq_length'])
              for layer_dec_cross_attn in decoder_cross_attn_probs)

if __name__ == '__main__':
    pytest.main([__file__])
