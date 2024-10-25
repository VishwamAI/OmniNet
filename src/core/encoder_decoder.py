"""
Encoder and Decoder implementation for the OmniNet transformer architecture.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base_layers import (
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding
)


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward networks.
    Includes residual connections and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: str = "gelu",
        device_config: Optional[dict] = None,
        head_pruning: bool = False,
        use_memory_efficient_attention: bool = False,
        **kwargs  # Accept and ignore extra parameters
    ):
        super().__init__()
        self.device_config = device_config or {}
        self.head_pruning = head_pruning
        self.use_memory_efficient_attention = use_memory_efficient_attention

        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            device_config=self.device_config,
            head_pruning=head_pruning,
            use_memory_efficient_attention=use_memory_efficient_attention
        )
        self.attention_norm = LayerNorm(hidden_size)
        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout=hidden_dropout,
            device_config=self.device_config
        )
        self.feedforward_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        normed_hidden_states = self.attention_norm(hidden_states)
        attention_output, attention_probs = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask
        )
        attention_output = self.dropout(attention_output)
        hidden_states = hidden_states + attention_output

        # Feed-forward with residual connection
        normed_hidden_states = self.feedforward_norm(hidden_states)
        feedforward_output = self.feedforward(normed_hidden_states)
        feedforward_output = self.dropout(feedforward_output)
        layer_output = hidden_states + feedforward_output

        return layer_output, attention_probs


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with masked self-attention, cross-attention,
    and feed-forward networks. Includes residual connections and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: str = "gelu",
        device_config: Optional[dict] = None,
        head_pruning: bool = False,
        use_memory_efficient_attention: bool = False,
        **kwargs  # Accept and ignore extra parameters
    ):
        super().__init__()
        self.device_config = device_config or {}
        self.head_pruning = head_pruning
        self.use_memory_efficient_attention = use_memory_efficient_attention

        # Self-attention
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            device_config=self.device_config,
            head_pruning=head_pruning,
            use_memory_efficient_attention=use_memory_efficient_attention
        )
        self.self_attention_norm = LayerNorm(hidden_size)

        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            device_config=self.device_config,
            head_pruning=head_pruning,
            use_memory_efficient_attention=use_memory_efficient_attention
        )
        self.cross_attention_norm = LayerNorm(hidden_size)

        # Feed-forward
        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            hidden_dropout=hidden_dropout,
            device_config=self.device_config
        )
        self.feedforward_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        normed_hidden = self.self_attention_norm(hidden_states)
        self_attention_output, self_attention_probs = self.self_attention(
            normed_hidden,
            attention_mask=self_attention_mask
        )
        self_attention_output = self.dropout(self_attention_output)
        hidden_states = hidden_states + self_attention_output

        # Cross-attention with residual connection
        normed_hidden = self.cross_attention_norm(hidden_states)
        cross_attention_output, cross_attention_probs = self.cross_attention(
            normed_hidden,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_attention_mask
        )
        cross_attention_output = self.dropout(cross_attention_output)
        hidden_states = hidden_states + cross_attention_output

        # Feed-forward with residual connection
        normed_hidden = self.feedforward_norm(hidden_states)
        feedforward_output = self.feedforward(normed_hidden)
        feedforward_output = self.dropout(feedforward_output)
        layer_output = hidden_states + feedforward_output

        return layer_output, self_attention_probs, cross_attention_probs


class Encoder(nn.Module):
    """
    Transformer encoder with multiple encoder layers and positional encoding.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 512,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: str = "gelu",
        device_config: Optional[dict] = None,
        head_pruning: bool = False,
        use_memory_efficient_attention: bool = False,
        **kwargs  # Accept and ignore extra parameters
    ):
        super().__init__()
        self.device_config = device_config or {}
        self.head_pruning = head_pruning
        self.use_memory_efficient_attention = use_memory_efficient_attention

        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=hidden_dropout
        )

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                activation=activation,
                device_config=self.device_config,
                head_pruning=head_pruning,
                use_memory_efficient_attention=use_memory_efficient_attention
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add positional encoding
        hidden_states = self.positional_encoding(hidden_states)

        # Process through encoder layers
        all_attention_probs = []
        for layer in self.layers:
            hidden_states, attention_probs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            all_attention_probs.append(attention_probs)

        return hidden_states, torch.stack(all_attention_probs)


class Decoder(nn.Module):
    """
    Transformer decoder with multiple decoder layers and positional encoding.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int = 512,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: str = "gelu",
        device_config: Optional[dict] = None,
        head_pruning: bool = False,
        use_memory_efficient_attention: bool = False,
        **kwargs  # Accept and ignore extra parameters
    ):
        super().__init__()
        self.device_config = device_config or {}
        self.head_pruning = head_pruning
        self.use_memory_efficient_attention = use_memory_efficient_attention

        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=hidden_dropout
        )

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                activation=activation,
                device_config=self.device_config,
                head_pruning=head_pruning,
                use_memory_efficient_attention=use_memory_efficient_attention
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Add positional encoding
        hidden_states = self.positional_encoding(hidden_states)

        # Process through decoder layers
        all_self_attention_probs = []
        all_cross_attention_probs = []

        for layer in self.layers:
            hidden_states, self_attention_probs, cross_attention_probs = layer(
                hidden_states,
                encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask
            )
            all_self_attention_probs.append(self_attention_probs)
            all_cross_attention_probs.append(cross_attention_probs)

        return (
            hidden_states,
            torch.stack(all_self_attention_probs),
            torch.stack(all_cross_attention_probs)
        )


class TransformerModel(nn.Module):
    """
    Complete transformer model with encoder and decoder.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        vocab_size: int = 30000,
        max_position_embeddings: int = 512,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: str = "gelu",
        head_pruning: bool = False,
        use_memory_efficient_attention: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Input embeddings
        self.encoder_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            head_pruning=head_pruning,
            use_memory_efficient_attention=use_memory_efficient_attention
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            head_pruning=head_pruning,
            use_memory_efficient_attention=use_memory_efficient_attention
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle both old and new parameter names
        encoder_input = encoder_input if encoder_input is not None else input_ids
        decoder_input = decoder_input if decoder_input is not None else decoder_input_ids
        encoder_attention_mask = encoder_attention_mask if encoder_attention_mask is not None else attention_mask

        # Apply input embeddings
        if encoder_input is not None:
            encoder_hidden_states = self.encoder_embeddings(encoder_input)
        else:
            raise ValueError("No input provided to encoder")

        if decoder_input is not None:
            decoder_hidden_states = self.decoder_embeddings(decoder_input)
        else:
            raise ValueError("No input provided to decoder")

        # Encode input sequence
        encoder_output, encoder_attention_probs = self.encoder(
            encoder_hidden_states,
            attention_mask=encoder_attention_mask
        )

        # Decode output sequence
        decoder_output, decoder_self_attention_probs, decoder_cross_attention_probs = (
            self.decoder(
                decoder_hidden_states,
                encoder_output,
                self_attention_mask=decoder_attention_mask,
                cross_attention_mask=cross_attention_mask
            )
        )

        return (
            decoder_output,
            encoder_attention_probs,
            decoder_self_attention_probs,
            decoder_cross_attention_probs
        )
