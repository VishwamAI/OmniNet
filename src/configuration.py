from typing import Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class OmniNetConfig:
    """Configuration class for OmniNet transformer model.
    
    Args:
        hidden_size (int): Size of the hidden layers
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        dropout (float): Dropout rate
        max_position_embeddings (int): Maximum sequence length
        use_memory_efficient_attention (bool): Whether to use memory-efficient attention
        device_config (Dict): Hardware-specific configuration
    """
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.1
    max_position_embeddings: int = 2048
    use_memory_efficient_attention: bool = True
    device_config: Dict[str, Union[str, float]] = None

    def __post_init__(self):
        if self.device_config is None:
            self.device_config = {
                'device': 'cpu',
                'precision': 'fp32'
            }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'OmniNetConfig':
        """Creates a config from a Python dictionary of parameters."""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Serializes this instance to a Python dictionary."""
        output = {}
        for field in self.__dataclass_fields__:
            output[field] = getattr(self, field)
        return output

    def save_pretrained(self, save_directory: str):
        """Save a configuration object to the directory."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, "config.json")
        
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> 'OmniNetConfig':
        """Load a configuration object from a directory."""
        import os
        import json
        
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, "r", encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls.from_dict(config_dict)
