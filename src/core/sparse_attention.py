import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

class SparseAttention(nn.Module):
    """Sparse Attention implementation with block-sparse patterns and memory awareness."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        sparsity_factor: float = 0.9,
        attention_dropout: float = 0.1,
        use_memory_efficient: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparsity_factor = sparsity_factor
        self.attention_dropout = attention_dropout
        self.use_memory_efficient = use_memory_efficient
        
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        
    def _compute_block_mask(self, seq_len: int) -> torch.Tensor:
        """Compute block-sparse attention mask."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        rand_mask = torch.rand(num_blocks, num_blocks)
        sparse_mask = (rand_mask > self.sparsity_factor).float()
        
        # Ensure causal masking for autoregressive models
        causal_mask = torch.triu(torch.ones(num_blocks, num_blocks))
        block_mask = sparse_mask * causal_mask
        
        # Expand mask to full sequence length
        full_mask = block_mask.repeat_interleave(self.block_size, dim=0)
        full_mask = full_mask.repeat_interleave(self.block_size, dim=1)
        return full_mask[:seq_len, :seq_len]
    
    def _memory_efficient_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention with memory efficiency using chunked computation."""
        batch_size, num_heads, seq_len, head_dim = query.size()
        
        # Compute attention scores in chunks
        chunk_size = min(self.block_size, seq_len)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        output = torch.zeros_like(query)
        
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, seq_len)
            
            # Current chunk of query
            q_chunk = query[:, :, chunk_start:chunk_end]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * self.scaling
            
            if mask is not None:
                chunk_mask = mask[:, :, chunk_start:chunk_end, :]
                scores = scores.masked_fill(~chunk_mask, float('-inf'))
            
            # Apply softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Compute chunk output
            chunk_output = torch.matmul(attn_weights, value)
            output[:, :, chunk_start:chunk_end] = chunk_output
            
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute block-sparse attention mask
        if attention_mask is None:
            attention_mask = self._compute_block_mask(seq_len).to(hidden_states.device)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)
        
        # Apply layer head mask if provided
        if layer_head_mask is not None:
            attention_mask = attention_mask * layer_head_mask.unsqueeze(-1).unsqueeze(-1)
        
        # Compute attention with memory efficiency
        if self.use_memory_efficient:
            attn_output = self._memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attention_mask
            )
        else:
            # Standard attention computation
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attn_output = torch.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attention_mask
