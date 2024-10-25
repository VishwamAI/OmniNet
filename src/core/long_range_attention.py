import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class LinformerAttention(nn.Module):
    """Linformer attention implementation for efficient long-range dependencies."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        seq_len: int,
        linformer_k: int = 256,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.linformer_k = min(linformer_k, seq_len)
        
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Linformer projections
        self.E = nn.Parameter(torch.Tensor(self.num_heads, self.linformer_k, seq_len))
        self.F = nn.Parameter(torch.Tensor(self.num_heads, self.linformer_k, seq_len))
        
        self.dropout = nn.Dropout(attention_dropout)
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize Linformer projection matrices."""
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)
        
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
        
        # Project key and value using Linformer projections
        projected_keys = torch.matmul(self.E.unsqueeze(0), key_states)
        projected_values = torch.matmul(self.F.unsqueeze(0), value_states)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, projected_keys.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            # Project attention mask
            projected_mask = torch.matmul(self.E.unsqueeze(0), attention_mask.transpose(-2, -1))
            attention_scores = attention_scores.masked_fill(~projected_mask, float('-inf'))
        
        # Apply layer head mask if provided
        if layer_head_mask is not None:
            attention_scores = attention_scores * layer_head_mask.unsqueeze(-1)
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        attn_output = torch.matmul(attention_probs, projected_values)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attention_probs
