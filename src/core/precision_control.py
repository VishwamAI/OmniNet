import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List
import psutil
import GPUtil
import numpy as np

class AdaptivePrecisionController:
    """Controls precision across different layers based on sensitivity and resource usage."""

    def __init__(
        self,
        num_layers: int,
        device_config: Dict[str, Union[str, float]],
        critical_layers: List[int] = None,
        memory_threshold: float = 0.85,
        initial_precision: str = 'fp32'
    ):
        self.num_layers = num_layers
        self.device_config = device_config
        self.memory_threshold = memory_threshold
        self.initial_precision = initial_precision

        # Define critical layers (embeddings, final layers by default)
        self.critical_layers = critical_layers or [0, num_layers-1]

        # Initialize layer-specific precision settings
        self.layer_precision = {}
        self._initialize_precision()

        # Track resource usage history
        self.resource_history = []

    def _initialize_precision(self):
        """Initialize precision settings for each layer."""
        for layer_idx in range(self.num_layers):
            if layer_idx in self.critical_layers:
                # Critical layers use higher precision
                self.layer_precision[layer_idx] = 'fp32'
            else:
                # Non-critical layers start with initial precision
                self.layer_precision[layer_idx] = self.initial_precision

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage ratio based on device type."""
        device = self.device_config.get('device', 'cpu')

        if device == 'cpu':
            return psutil.virtual_memory().percent / 100.0
        elif device == 'cuda':
            try:
                gpu = GPUtil.getGPUs()[0]  # Get first GPU
                return gpu.memoryUsed / gpu.memoryTotal
            except:
                return 0.0
        return 0.0

    def _adjust_precision_based_on_memory(self):
        """Dynamically adjust precision based on memory usage."""
        current_usage = self._get_current_memory_usage()
        self.resource_history.append(current_usage)

        # Keep only recent history
        if len(self.resource_history) > 10:
            self.resource_history.pop(0)

        # Calculate trend
        trend_increasing = len(self.resource_history) > 1 and \
                         np.mean(np.diff(self.resource_history)) > 0

        if current_usage > self.memory_threshold or trend_increasing:
            # Reduce precision for non-critical layers
            for layer_idx in range(self.num_layers):
                if layer_idx not in self.critical_layers:
                    if self.layer_precision[layer_idx] == 'fp32':
                        self.layer_precision[layer_idx] = 'fp16'
                    elif self.layer_precision[layer_idx] == 'fp16':
                        self.layer_precision[layer_idx] = 'int8'
        elif current_usage < self.memory_threshold * 0.7:
            # Increase precision if memory usage is low
            for layer_idx in range(self.num_layers):
                if layer_idx not in self.critical_layers:
                    if self.layer_precision[layer_idx] == 'int8':
                        self.layer_precision[layer_idx] = 'fp16'
                    elif self.layer_precision[layer_idx] == 'fp16':
                        self.layer_precision[layer_idx] = 'fp32'

    def get_layer_precision(self, layer_idx: int) -> str:
        """Get current precision for a specific layer."""
        self._adjust_precision_based_on_memory()
        return self.layer_precision.get(layer_idx, self.initial_precision)

class PrecisionAdaptiveLayer(nn.Module):
    """Layer wrapper that handles precision adaptation."""

    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        precision_controller: AdaptivePrecisionController
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.precision_controller = precision_controller

    def _cast_inputs(self, *args, precision: str):
        """Cast inputs to specified precision."""
        if precision == 'fp32':
            return [x.float() if torch.is_tensor(x) else x for x in args]
        elif precision == 'fp16':
            return [x.half() if torch.is_tensor(x) else x for x in args]
        elif precision == 'int8':
            # Quantize to int8 with dynamic range
            def quantize(x):
                if not torch.is_tensor(x):
                    return x
                with torch.no_grad():
                    scale = x.abs().max() / 127.0
                    return (x / scale).round().clamp(-128, 127).char(), scale
            return [quantize(x) if torch.is_tensor(x) else x for x in args]
        return args

    def _cast_output(self, output, precision: str, input_scale: Optional[torch.Tensor] = None):
        """Cast output back to appropriate precision."""
        if precision == 'int8' and input_scale is not None:
            if isinstance(output, tuple):
                return tuple(x * input_scale if torch.is_tensor(x) else x for x in output)
            return output * input_scale
        return output

    def forward(self, *args, **kwargs):
        """Forward pass with precision adaptation."""
        precision = self.precision_controller.get_layer_precision(self.layer_idx)

        # Cast inputs to appropriate precision
        cast_args = self._cast_inputs(*args, precision=precision)
        input_scale = None
        if precision == 'int8':
            cast_args, input_scale = zip(*[
                (x[0], x[1]) if isinstance(x, tuple) and len(x) == 2
                else (x, None) for x in cast_args
            ])
            input_scale = next((s for s in input_scale if s is not None), None)

        # Forward pass
        output = self.layer(*cast_args, **kwargs)

        # Cast output back
        return self._cast_output(output, precision, input_scale)
