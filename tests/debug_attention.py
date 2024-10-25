import torch
import torch.nn as nn
from src.core.base_layers import MultiHeadAttention

def test_debug_causal_attention():
    # Create a small model for testing
    attention = MultiHeadAttention(
        hidden_size=256,
        num_attention_heads=8,
        attention_dropout=0.1
    )

    # Create sample inputs
    batch_size = 2
    seq_length = 8
    hidden_size = 256
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)

    # Create causal mask
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    causal_mask = -10000.0 * mask.unsqueeze(0)  # [1, seq_length, seq_length]
    # Expand for batch size and attention heads
    causal_mask = causal_mask.expand(batch_size, -1, -1)  # [batch_size, seq_length, seq_length]

    print("\nInput shapes:")
    print(f"hidden_states: {hidden_states.shape}")
    print(f"causal_mask: {causal_mask.shape}")

    # Forward pass with shape tracking
    query_layer = attention.transpose_for_scores(attention.query(hidden_states))
    key_layer = attention.transpose_for_scores(attention.key(hidden_states))
    value_layer = attention.transpose_for_scores(attention.value(hidden_states))

    print("\nTransformed shapes:")
    print(f"query_layer: {query_layer.shape}")
    print(f"key_layer: {key_layer.shape}")
    print(f"value_layer: {value_layer.shape}")

    # Try the forward pass
    try:
        output, attention_probs = attention(hidden_states, attention_mask=causal_mask)
        print("\nSuccess!")
        print(f"output shape: {output.shape}")
        print(f"attention_probs shape: {attention_probs.shape}")
    except Exception as e:
        print("\nError occurred:")
        print(str(e))

if __name__ == "__main__":
    test_debug_causal_attention()
