import torch
import torch.nn as nn
import math

class LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__(hidden_size, eps=eps)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.1, head_pruning=False, device_config=None, use_memory_efficient_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.active_heads = num_attention_heads
        self.device_config = device_config or {}

        # Initialize with proper device and dtype based on config
        device = self.device_config.get('device', 'cpu')
        dtype = torch.float32 if self.device_config.get('precision', 'fp32') == 'fp32' else torch.float16

        self.query = nn.Linear(hidden_size, self.all_head_size).to(device=device, dtype=dtype)
        self.key = nn.Linear(hidden_size, self.all_head_size).to(device=device, dtype=dtype)
        self.value = nn.Linear(hidden_size, self.all_head_size).to(device=device, dtype=dtype)

        self.dropout = nn.Dropout(attention_dropout)
        self.head_pruning = head_pruning
        self.head_importance = None
        self.head_weights = nn.Parameter(torch.ones(num_attention_heads, device=device, dtype=dtype)) if head_pruning else None
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.key_cache = {}
        self.value_cache = {}

    def set_active_heads(self, num_heads):
        """Dynamically adjust the number of active attention heads."""
        assert 0 < num_heads <= self.num_attention_heads
        self.active_heads = num_heads

    def compute_head_importance(self, hidden_states):
        """Compute importance scores for each attention head."""
        with torch.no_grad():
            # Get attention outputs for each head separately
            batch_size, seq_length = hidden_states.size()[:2]
            query = self.transpose_for_scores(self.query(hidden_states))
            key = self.transpose_for_scores(self.key(hidden_states))
            value = self.transpose_for_scores(self.value(hidden_states))

            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # Compute importance as the average attention weight per head
            head_importance = attention_probs.mean(dim=[0, 2, 3])  # Average across batch, seq_len dimensions
            self.head_importance = head_importance
            return head_importance

    def prune_heads(self, num_heads_to_prune):
        """Disable the least important attention heads."""
        if self.head_importance is None:
            raise ValueError("Must compute head importance before pruning")

        _, indices = torch.sort(self.head_importance)
        heads_to_prune = indices[:num_heads_to_prune]
        self.head_weights.data[heads_to_prune] = 0.0
        self.active_heads = self.num_attention_heads - num_heads_to_prune

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None):
        batch_size, seq_length = hidden_states.size()[:2]

        # Generate unique cache key based on input tensor properties
        cache_key = f"{hidden_states.shape}_{hidden_states.device}"

        # For cross-attention, use encoder_hidden_states for key and value
        key_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        value_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Convert attention mask to dense tensor if it's sparse
        if attention_mask is not None and attention_mask.is_sparse:
            attention_mask = attention_mask.to_dense()

        if self.use_memory_efficient_attention and cache_key in self.key_cache:
            key_layer = self.key_cache[cache_key]
            value_layer = self.value_cache[cache_key]
        else:
            key_layer = self.transpose_for_scores(self.key(key_states))
            value_layer = self.transpose_for_scores(self.value(value_states))
            if self.use_memory_efficient_attention:
                self.key_cache[cache_key] = key_layer
                self.value_cache[cache_key] = value_layer

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Use only active heads
        query_layer = query_layer[:, :self.active_heads]
        key_layer = key_layer[:, :self.active_heads]
        value_layer = value_layer[:, :self.active_heads]

        if self.use_memory_efficient_attention:
            # Implement flash attention pattern: compute attention in chunks
            chunk_size = min(128, seq_length)  # Adjust chunk size based on sequence length
            num_chunks = (seq_length + chunk_size - 1) // chunk_size

            # Initialize context layer with correct dimensions
            context_layer = torch.zeros(
                batch_size,
                self.active_heads,
                seq_length,
                self.attention_head_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, seq_length)

                # Compute attention scores for current chunk
                chunk_query = query_layer[:, :, start_idx:end_idx]
                chunk_key = key_layer

                chunk_scores = torch.matmul(
                    chunk_query,
                    chunk_key.transpose(-1, -2)
                ) / math.sqrt(self.attention_head_size)

                if attention_mask is not None:
                    # Properly reshape mask for chunked attention
                    chunk_mask = attention_mask[:, start_idx:end_idx, :seq_length]  # Include full sequence length
                    chunk_mask = chunk_mask.unsqueeze(1).expand(-1, self.active_heads, -1, -1)
                    chunk_scores = chunk_scores + chunk_mask

                chunk_probs = nn.functional.softmax(chunk_scores, dim=-1)
                chunk_probs = self.dropout(chunk_probs)

                if self.head_pruning:
                    chunk_probs = chunk_probs * self.head_weights[:self.active_heads].view(1, -1, 1, 1)

                # Update context layer for current chunk
                context_layer[:, :, start_idx:end_idx] = torch.matmul(chunk_probs, value_layer)

            output_attention_probs = chunk_probs  # Store last chunk's attention probs
        else:
            # Original attention implementation
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                # Reshape attention mask to match attention scores dimensions [batch, heads, query_seq, key_seq]
                attention_mask = attention_mask.unsqueeze(1).expand(-1, self.active_heads, -1, -1)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            if self.head_pruning:
                attention_probs = attention_probs * self.head_weights[:self.active_heads].view(1, -1, 1, 1)

            output_attention_probs = attention_probs
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Pad inactive heads with zeros to maintain consistent output dimension
        if self.active_heads < self.num_attention_heads:
            pad_size = self.num_attention_heads - self.active_heads
            zeros = torch.zeros_like(context_layer[:, :, :pad_size, :])
            context_layer = torch.cat([context_layer, zeros], dim=2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, output_attention_probs

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout=0.1, activation='gelu', device_config=None):
        super().__init__()
        self.device_config = device_config or {}

        # Initialize with proper device and dtype based on config
        device = self.device_config.get('device', 'cpu')
        dtype = torch.float32 if self.device_config.get('precision', 'fp32') == 'fp32' else torch.float16

        self.dense1 = nn.Linear(hidden_size, intermediate_size).to(device=device, dtype=dtype)
        self.dense2 = nn.Linear(intermediate_size, hidden_size).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(hidden_dropout)

        activation = activation.lower()
        if activation == 'gelu':
            self.intermediate_act_fn = nn.GELU()
        elif activation == 'relu':
            self.intermediate_act_fn = nn.ReLU()
        elif activation == 'silu':
            self.intermediate_act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_position_embeddings, hidden_size)
        )

    def forward(self, x):
        position_embeddings = self.position_embeddings[:, :x.size(1), :]
        x = x + position_embeddings
        return self.dropout(x)
