import pytest
import torch
import torch.nn as nn

from src.core.base_layers import MultiHeadAttention
from src.core.encoder_decoder import EncoderLayer
from src.hardware.optimizations import get_device_config

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

def test_dynamic_attention_heads(model_config):
    """Test if attention heads can be dynamically adjusted"""
    attention = MultiHeadAttention(
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        attention_dropout=model_config['attention_dropout']
    )
    
    x = torch.randn(model_config['batch_size'], model_config['seq_length'], model_config['hidden_size'])
    
    # Test with different numbers of active heads
    active_heads = [4, 6, 8]
    for num_heads in active_heads:
        attention.set_active_heads(num_heads)
        output, attention_probs = attention(x)
        
        # Verify output shape remains consistent
        assert output.shape == x.shape
        # Verify only specified number of heads are active
        assert attention_probs.shape == (model_config['batch_size'], num_heads, 
                                      model_config['seq_length'], model_config['seq_length'])

def test_adaptive_head_pruning(model_config):
    """Test if attention heads can be pruned based on importance"""
    attention = MultiHeadAttention(
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        attention_dropout=model_config['attention_dropout'],
        head_pruning=True
    )

    x = torch.randn(model_config['batch_size'], model_config['seq_length'], model_config['hidden_size'])

    # Get head importance scores
    head_importance = attention.compute_head_importance(x)
    assert head_importance.shape == (model_config['num_attention_heads'],)

    # Test pruning least important heads
    attention.prune_heads(2)  # Prune 2 least important heads
    output, attention_probs = attention(x)

    # Verify output shape and reduced number of attention heads
    assert output.shape == x.shape
    assert attention_probs.shape == (model_config['batch_size'],
                                   model_config['num_attention_heads'] - 2,
                                   model_config['seq_length'],
                                   model_config['seq_length'])

def test_hardware_compatibility(model_config):
    """Test model compatibility across different hardware configurations"""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        config = get_device_config(device)
        
        # Adjust model configuration based on hardware
        encoder_layer = EncoderLayer(
            hidden_size=model_config['hidden_size'],
            num_attention_heads=model_config['num_attention_heads'],
            intermediate_size=model_config['intermediate_size'],
            attention_dropout=model_config['attention_dropout'],
            hidden_dropout=model_config['hidden_dropout'],
            device_config=config
        )
        
        x = torch.randn(model_config['batch_size'], 
                       model_config['seq_length'],
                       model_config['hidden_size']).to(device)
        
        # Test forward pass on device
        output, _ = encoder_layer(x)
        assert output.device.type == device
        assert output.shape == x.shape

def test_extended_context(model_config):
    """Test handling of longer sequences"""
    attention = MultiHeadAttention(
        hidden_size=model_config['hidden_size'],
        num_attention_heads=model_config['num_attention_heads'],
        attention_dropout=model_config['attention_dropout']
    )
    
    # Test with sequence length longer than standard
    long_seq_length = 1024
    x = torch.randn(model_config['batch_size'], long_seq_length, model_config['hidden_size'])
    
    output, attention_probs = attention(x)
    
    # Verify handling of longer sequence
    assert output.shape == x.shape
    assert attention_probs.shape == (model_config['batch_size'], 
                                   model_config['num_attention_heads'],
                                   long_seq_length, long_seq_length)

if __name__ == '__main__':
    pytest.main([__file__])
