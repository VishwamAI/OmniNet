"""
Tests for hardware-specific optimizations of the OmniNet transformer.
"""

import pytest
import torch
import torch.nn as nn

from src.hardware.optimizations import (
    HardwareConfig,
    HardwareOptimizer,
    OptimizedTransformer
)
from src.core.encoder_decoder import TransformerModel


@pytest.fixture
def base_model_config():
    return {
        'num_layers': 2,
        'hidden_size': 256,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'attention_dropout': 0.1,
        'hidden_dropout': 0.1
    }


@pytest.fixture
def sample_input():
    batch_size = 4
    seq_length = 16
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'decoder_input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'decoder_attention_mask': torch.ones(batch_size, seq_length)
    }


class TestHardwareConfig:
    """Test suite for HardwareConfig."""

    def test_config_initialization(self):
        config = HardwareConfig(
            device_type="gpu",
            precision="fp16",
            memory_efficient=True,
            use_flash_attention=True
        )
        assert config.device_type == "gpu"
        assert config.precision == "fp16"
        assert config.memory_efficient
        assert config.use_flash_attention

    def test_dtype_mapping(self):
        configs = [
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            ("int8", torch.int8)
        ]
        for precision, expected_dtype in configs:
            config = HardwareConfig(precision=precision)
            assert config.dtype == expected_dtype


class TestHardwareOptimizer:
    """Test suite for HardwareOptimizer."""

    @pytest.fixture
    def base_model(self, base_model_config):
        return TransformerModel(**base_model_config)

    def test_cpu_optimization(self, base_model):
        config = HardwareConfig(
            device_type="cpu",
            precision="fp32",
            quantization="dynamic"
        )
        optimizer = HardwareOptimizer(config)
        optimized_model = optimizer.optimize_model(base_model)
        
        # Verify model is on CPU with correct dtype
        assert next(optimized_model.parameters()).device.type == "cpu"
        assert next(optimized_model.parameters()).dtype == torch.float32
        
        # Check if quantization was applied
        assert hasattr(optimized_model, "qconfig")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_optimization(self, base_model):
        config = HardwareConfig(
            device_type="gpu",
            precision="fp16",
            memory_efficient=True,
            use_flash_attention=True
        )
        optimizer = HardwareOptimizer(config)
        optimized_model = optimizer.optimize_model(base_model)
        
        # Verify model is on GPU with correct dtype
        assert next(optimized_model.parameters()).device.type == "cuda"
        assert next(optimized_model.parameters()).dtype == torch.float16
        
        # Check if gradient checkpointing is enabled
        assert optimized_model.is_gradient_checkpointing_enabled()

    def test_edge_device_optimization(self, base_model):
        config = HardwareConfig(
            device_type="npu",
            precision="int8",
            quantization="dynamic",
            memory_efficient=True
        )
        optimizer = HardwareOptimizer(config)
        optimized_model = optimizer.optimize_model(base_model)
        
        # Verify quantization was applied
        assert hasattr(optimized_model, "qconfig")
        
        # Check if gradient checkpointing is enabled
        assert optimized_model.is_gradient_checkpointing_enabled()


class TestOptimizedTransformer:
    """Test suite for OptimizedTransformer."""

    def test_model_creation(self, base_model_config):
        hardware_config = {
            "device_type": "cpu",
            "precision": "fp32",
            "memory_efficient": True,
            "use_flash_attention": False
        }
        
        optimized_model = OptimizedTransformer.create_optimized_model(
            TransformerModel(**base_model_config),
            hardware_config
        )
        
        assert isinstance(optimized_model, OptimizedTransformer)
        assert optimized_model.config.device_type == "cpu"
        assert optimized_model.config.precision == "fp32"

    def test_forward_pass(self, base_model_config, sample_input):
        hardware_config = {
            "device_type": "cpu",
            "precision": "fp32",
            "memory_efficient": True
        }
        
        optimized_model = OptimizedTransformer.create_optimized_model(
            TransformerModel(**base_model_config),
            hardware_config
        )
        
        outputs = optimized_model(**sample_input)
        assert outputs.shape == (
            sample_input['input_ids'].size(0),
            sample_input['input_ids'].size(1),
            base_model_config['hidden_size']
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, base_model_config, sample_input):
        hardware_config = {
            "device_type": "gpu",
            "precision": "fp16",
            "memory_efficient": True
        }
        
        optimized_model = OptimizedTransformer.create_optimized_model(
            TransformerModel(**base_model_config),
            hardware_config
        )
        
        # Verify mixed precision setup
        assert next(optimized_model.parameters()).dtype == torch.float16
        
        # Test forward pass with mixed precision
        outputs = optimized_model(**sample_input)
        assert outputs.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__])
