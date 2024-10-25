import torch
import torch.nn as nn
from src.core.base_layers import MultiHeadAttention, LayerNorm, FeedForward, PositionalEncoding

def test_dynamic_attention():
    """Test dynamic attention heads and adaptive mechanism"""
    attention = MultiHeadAttention(
        hidden_size=256,
        num_attention_heads=8,
        attention_dropout=0.1,
        head_pruning=True  # Enable head pruning
    )

    # Test head importance computation
    batch_size = 2
    seq_length = 8
    hidden_size = 256
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)

    # Create attention mask for importance computation
    attention_mask = torch.zeros(batch_size, seq_length, seq_length)

    # Compute head importance with mask
    head_importance = attention.compute_head_importance(hidden_states)
    assert head_importance.shape == (8,), "Head importance shape mismatch"
    assert not torch.isnan(head_importance).any(), "Head importance contains NaN values"
    assert not torch.isinf(head_importance).any(), "Head importance contains infinite values"

    # Test head pruning
    attention.prune_heads(2)  # Prune 2 least important heads
    assert attention.active_heads == 6, "Head pruning failed"
    assert torch.sum(attention.head_weights == 0) == 2, "Expected 2 heads to be pruned"

    # Test forward pass with pruned heads
    output, attention_probs = attention(hidden_states, attention_mask)
    assert output.shape == (batch_size, seq_length, hidden_size), "Output shape mismatch after pruning"
    assert attention_probs.shape == (batch_size, 6, seq_length, seq_length), "Attention probs shape mismatch"


def test_layer_norm_residual():
    """Test layer normalization and residual connections"""
    layer_norm = LayerNorm(256)
    feed_forward = FeedForward(256, 1024)

    # Test input
    x = torch.randn(2, 8, 256)

    # Test layer norm
    norm_output = layer_norm(x)
    # Use larger tolerance for numerical stability
    assert torch.allclose(norm_output.mean(dim=-1), torch.zeros_like(norm_output.mean(dim=-1)), atol=2e-3)
    assert torch.allclose(norm_output.std(dim=-1), torch.ones_like(norm_output.std(dim=-1)), atol=2e-3)

    # Verify layer norm properties
    assert norm_output.size() == x.size(), "Layer norm output shape mismatch"

    # Test residual connection
    residual = x
    ff_output = feed_forward(x)
    output = layer_norm(ff_output + residual)
    assert output.shape == x.shape, "Residual connection shape mismatch"

def test_extended_context():
    """Test extended context support with longer sequences"""
    attention = MultiHeadAttention(
        hidden_size=256,
        num_attention_heads=8,
        attention_dropout=0.1
    )
    
    # Test with longer sequence
    batch_size = 2
    seq_length = 1024  # Extended context
    hidden_size = 256
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    # Create causal mask for longer sequence
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    causal_mask = -10000.0 * mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Test forward pass with longer sequence
    output, attention_probs = attention(hidden_states, attention_mask=causal_mask)
    assert output.shape == (batch_size, seq_length, hidden_size), "Extended context output shape mismatch"
    assert attention_probs.shape == (batch_size, 8, seq_length, seq_length), "Extended context attention probs shape mismatch"

def test_positional_encoding():
    """Test learnable positional encodings"""
    pos_encoding = PositionalEncoding(256, max_position_embeddings=2048)

    # Test different sequence lengths
    for seq_length in [8, 64, 512, 1024]:
        x = torch.randn(2, seq_length, 256)
        output = pos_encoding(x)
        assert output.shape == x.shape, f"Positional encoding shape mismatch for length {seq_length}"

def test_hardware_compatibility():
    """Test compatibility with different hardware configurations"""
    attention = MultiHeadAttention(
        hidden_size=256,
        num_attention_heads=8,
        attention_dropout=0.1
    )
    
    batch_size = 2
    seq_length = 8
    hidden_size = 256
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    # Test CPU
    output_cpu, _ = attention(hidden_states)
    assert output_cpu.device.type == "cpu", "CPU compatibility test failed"
    
    # Test GPU if available
    if torch.cuda.is_available():
        hidden_states_gpu = hidden_states.cuda()
        attention_gpu = attention.cuda()
        output_gpu, _ = attention_gpu(hidden_states_gpu)
        assert output_gpu.device.type == "cuda", "GPU compatibility test failed"
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            output_amp, _ = attention_gpu(hidden_states_gpu)
            assert output_amp.dtype == torch.float16, "Mixed precision test failed"

if __name__ == "__main__":
    print("\nTesting dynamic attention...")
    test_dynamic_attention()
    print("✓ Dynamic attention tests passed")
    
    print("\nTesting layer normalization and residual connections...")
    test_layer_norm_residual()
    print("✓ Layer norm and residual tests passed")
    
    print("\nTesting extended context support...")
    test_extended_context()
    print("✓ Extended context tests passed")
    
    print("\nTesting positional encoding...")
    test_positional_encoding()
    print("✓ Positional encoding tests passed")
    
    print("\nTesting hardware compatibility...")
    test_hardware_compatibility()
    print("✓ Hardware compatibility tests passed")
