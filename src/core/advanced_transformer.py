import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple

from .sparse_attention import SparseAttention
from .long_range_attention import LinformerAttention
from .precision_control import AdaptivePrecisionController, PrecisionAdaptiveLayer

class AdvancedTransformerLayer(nn.Module):
    """Advanced transformer layer with sparse attention and precision control."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        layer_idx: int = 0,
        max_seq_length: int = 2048,
        use_sparse_attention: bool = True,
        use_linformer: bool = True,
        device_config: Optional[Dict[str, Union[str, float]]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        
        # Initialize precision controller
        self.precision_controller = AdaptivePrecisionController(
            num_layers=1,  # Local to this layer
            device_config=device_config or {'device': 'cpu', 'precision': 'fp32'},
            critical_layers=[0] if layer_idx in [0, -1] else []
        )
        
        # Initialize attention mechanism
        if use_sparse_attention:
            self.attention = SparseAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                block_size=64,
                sparsity_factor=0.9,
                attention_dropout=dropout,
                use_memory_efficient=True
            )
        elif use_linformer:
            self.attention = LinformerAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                seq_len=max_seq_length,
                linformer_k=min(256, max_seq_length // 4),
                attention_dropout=dropout
            )
        
        # Wrap attention with precision control
        self.attention = PrecisionAdaptiveLayer(
            self.attention,
            layer_idx,
            self.precision_controller
        )
        
        # Layer normalization and feedforward
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Wrap feedforward with precision control
        self.feedforward = PrecisionAdaptiveLayer(
            self.feedforward,
            layer_idx,
            self.precision_controller
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply layer norm first (Pre-LN transformer architecture)
        normed_hidden_states = self.layer_norm1(hidden_states)
        
        # Self-attention
        attention_output, attention_weights = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # Feedforward
        normed_hidden_states = self.layer_norm2(hidden_states)
        feedforward_output = self.feedforward(normed_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + feedforward_output
        
        return hidden_states, attention_weights

class AdvancedTransformer(nn.Module):
    """Advanced transformer with sparse attention, precision control, and hardware optimization."""
    
    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 1024,
        num_heads: int = 16,
        ff_dim: int = 4096,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        use_sparse_attention: bool = True,
        use_linformer: bool = True,
        device_config: Optional[Dict[str, Union[str, float]]] = None,
    ):
        super().__init__()
        
        # Initialize device configuration
        self.device_config = device_config or {'device': 'cpu', 'precision': 'fp32'}
        
        # Initialize precision controller for the whole model
        self.precision_controller = AdaptivePrecisionController(
            num_layers=num_layers,
            device_config=self.device_config,
            critical_layers=[0, num_layers-1]  # Embeddings and final layer
        )
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            AdvancedTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                layer_idx=i,
                max_seq_length=max_seq_length,
                use_sparse_attention=use_sparse_attention,
                use_linformer=use_linformer,
                device_config=self.device_config
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_hidden_states = []
        all_attention_weights = []
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
            )
            
            all_hidden_states.append(hidden_states)
            all_attention_weights.append(attention_weights)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states, all_attention_weights
