"""
GPU Acceleration Module for Name Matching

This module provides GPU-accelerated implementations of computationally intensive
name matching operations using multiple GPU frameworks with automatic fallback.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import time
from functools import wraps
import warnings

logger = logging.getLogger(__name__)

# GPU Framework Detection and Imports
class GPUFramework:
    """Manages GPU framework availability and selection."""

    def __init__(self):
        self.cupy_available = False
        self.numba_cuda_available = False
        self.torch_available = False
        self.preferred_framework = None
        self.gpu_device_count = 0

        self._detect_frameworks()
        self._select_preferred_framework()

    def _detect_frameworks(self):
        """Detect available GPU frameworks."""

        # Check CuPy
        try:
            import cupy as cp
            if cp.cuda.is_available():
                self.cupy_available = True
                self.gpu_device_count = cp.cuda.runtime.getDeviceCount()
                logger.info(f"CuPy available with {self.gpu_device_count} GPU(s)")
            else:
                logger.info("CuPy installed but no CUDA devices available")
        except ImportError:
            logger.debug("CuPy not available")
        except Exception as e:
            logger.warning(f"CuPy detection failed: {e}")

        # Check Numba CUDA
        try:
            from numba import cuda
            if cuda.is_available():
                self.numba_cuda_available = True
                devices = cuda.list_devices()
                self.gpu_device_count = max(self.gpu_device_count, len(devices))
                logger.info(f"Numba CUDA available with {len(devices)} GPU(s)")
            else:
                logger.info("Numba CUDA installed but no CUDA devices available")
        except ImportError:
            logger.debug("Numba CUDA not available")
        except Exception as e:
            logger.warning(f"Numba CUDA detection failed: {e}")

        # Check PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                self.torch_available = True
                device_count = torch.cuda.device_count()
                self.gpu_device_count = max(self.gpu_device_count, device_count)
                logger.info(f"PyTorch CUDA available with {device_count} GPU(s)")
            else:
                logger.info("PyTorch installed but no CUDA devices available")
        except ImportError:
            logger.debug("PyTorch not available")
        except Exception as e:
            logger.warning(f"PyTorch detection failed: {e}")

    def _select_preferred_framework(self):
        """Select the preferred GPU framework based on availability and performance."""
        if self.cupy_available:
            self.preferred_framework = 'cupy'
            logger.info("Selected CuPy as preferred GPU framework")
        elif self.torch_available:
            self.preferred_framework = 'torch'
            logger.info("Selected PyTorch as preferred GPU framework")
        elif self.numba_cuda_available:
            self.preferred_framework = 'numba'
            logger.info("Selected Numba CUDA as preferred GPU framework")
        else:
            self.preferred_framework = None
            logger.info("No GPU frameworks available, using CPU fallback")

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU framework is available."""
        return self.preferred_framework is not None

    def get_framework_info(self) -> Dict[str, Any]:
        """Get detailed information about available frameworks."""
        return {
            'cupy_available': self.cupy_available,
            'numba_cuda_available': self.numba_cuda_available,
            'torch_available': self.torch_available,
            'preferred_framework': self.preferred_framework,
            'gpu_device_count': self.gpu_device_count,
            'has_gpu': self.has_gpu
        }

# Global GPU framework instance
gpu_framework = GPUFramework()

def gpu_fallback(cpu_func):
    """Decorator to provide automatic CPU fallback for GPU functions."""
    @wraps(cpu_func)
    def wrapper(*args, **kwargs):
        if not gpu_framework.has_gpu:
            return cpu_func(*args, **kwargs)

        try:
            # Try GPU implementation first
            gpu_func_name = f"{cpu_func.__name__}_gpu"
            if gpu_func_name in globals():
                return globals()[gpu_func_name](*args, **kwargs)
            else:
                logger.debug(f"GPU implementation {gpu_func_name} not found, using CPU")
                return cpu_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"GPU execution failed for {cpu_func.__name__}: {e}, falling back to CPU")
            return cpu_func(*args, **kwargs)

    return wrapper

class GPUStringMatcher:
    """GPU-accelerated string matching operations."""

    def __init__(self, framework: str = None):
        """
        Initialize GPU string matcher.

        Args:
            framework: Preferred GPU framework ('cupy', 'torch', 'numba', or None for auto)
        """
        self.framework = framework or gpu_framework.preferred_framework
        self.device_id = 0
        self._initialize_framework()

    def _initialize_framework(self):
        """Initialize the selected GPU framework."""
        if self.framework == 'cupy':
            try:
                import cupy as cp
                self.cp = cp
                cp.cuda.Device(self.device_id).use()
                logger.info(f"Initialized CuPy on device {self.device_id}")
            except Exception as e:
                logger.error(f"Failed to initialize CuPy: {e}")
                self.framework = None

        elif self.framework == 'torch':
            try:
                import torch
                self.torch = torch
                self.device = torch.device(f'cuda:{self.device_id}')
                logger.info(f"Initialized PyTorch on device {self.device}")
            except Exception as e:
                logger.error(f"Failed to initialize PyTorch: {e}")
                self.framework = None

        elif self.framework == 'numba':
            try:
                from numba import cuda
                self.cuda = cuda
                cuda.select_device(self.device_id)
                logger.info(f"Initialized Numba CUDA on device {self.device_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Numba CUDA: {e}")
                self.framework = None

    def batch_jaro_winkler_gpu(self, names1: List[str], names2: List[str],
                               prefix_weight: float = 0.1) -> np.ndarray:
        """
        GPU-accelerated batch Jaro-Winkler similarity calculation.

        Args:
            names1: First list of names
            names2: Second list of names
            prefix_weight: Jaro-Winkler prefix weight

        Returns:
            2D numpy array of similarity scores
        """
        if self.framework == 'cupy':
            return self._batch_jaro_winkler_cupy(names1, names2, prefix_weight)
        elif self.framework == 'torch':
            return self._batch_jaro_winkler_torch(names1, names2, prefix_weight)
        elif self.framework == 'numba':
            return self._batch_jaro_winkler_numba(names1, names2, prefix_weight)
        else:
            raise RuntimeError("No GPU framework available")

    def _batch_jaro_winkler_cupy(self, names1: List[str], names2: List[str],
                                 prefix_weight: float) -> np.ndarray:
        """CuPy implementation of batch Jaro-Winkler."""
        try:
            import cupy as cp

            # Convert strings to character arrays
            max_len1 = max(len(name) for name in names1) if names1 else 0
            max_len2 = max(len(name) for name in names2) if names2 else 0

            # Pad strings and convert to numeric arrays
            names1_padded = np.array([list(name.ljust(max_len1)[:max_len1]) for name in names1])
            names2_padded = np.array([list(name.ljust(max_len2)[:max_len2]) for name in names2])

            # Convert to ASCII codes
            names1_ascii = np.array([[ord(c) for c in name] for name in names1_padded])
            names2_ascii = np.array([[ord(c) for c in name] for name in names2_padded])

            # Transfer to GPU
            names1_gpu = cp.asarray(names1_ascii)
            names2_gpu = cp.asarray(names2_ascii)

            # Allocate result matrix
            result_gpu = cp.zeros((len(names1), len(names2)), dtype=cp.float32)

            # Custom CUDA kernel for Jaro-Winkler
            jaro_winkler_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void jaro_winkler_kernel(const int* names1, const int* names2,
                                   float* result, int n1, int n2, int max_len1, int max_len2,
                                   float prefix_weight) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                int j = blockIdx.y * blockDim.y + threadIdx.y;

                if (i >= n1 || j >= n2) return;

                // Simplified Jaro-Winkler calculation
                const int* s1 = names1 + i * max_len1;
                const int* s2 = names2 + j * max_len2;

                // Calculate actual string lengths
                int len1 = 0, len2 = 0;
                for (int k = 0; k < max_len1 && s1[k] != 32; k++) len1++;  // 32 = space
                for (int k = 0; k < max_len2 && s2[k] != 32; k++) len2++;

                if (len1 == 0 && len2 == 0) {
                    result[i * n2 + j] = 1.0f;
                    return;
                }
                if (len1 == 0 || len2 == 0) {
                    result[i * n2 + j] = 0.0f;
                    return;
                }

                // Jaro similarity calculation
                int match_window = max(len1, len2) / 2 - 1;
                if (match_window < 0) match_window = 0;

                bool* s1_matches = new bool[len1]();
                bool* s2_matches = new bool[len2]();

                int matches = 0;
                for (int k = 0; k < len1; k++) {
                    int start = max(0, k - match_window);
                    int end = min(k + match_window + 1, len2);

                    for (int l = start; l < end; l++) {
                        if (s2_matches[l] || s1[k] != s2[l]) continue;
                        s1_matches[k] = s2_matches[l] = true;
                        matches++;
                        break;
                    }
                }

                if (matches == 0) {
                    result[i * n2 + j] = 0.0f;
                    delete[] s1_matches;
                    delete[] s2_matches;
                    return;
                }

                // Count transpositions
                int transpositions = 0;
                int k = 0;
                for (int l = 0; l < len1; l++) {
                    if (!s1_matches[l]) continue;
                    while (!s2_matches[k]) k++;
                    if (s1[l] != s2[k]) transpositions++;
                    k++;
                }

                float jaro = (float(matches) / len1 + float(matches) / len2 +
                             float(matches - transpositions/2) / matches) / 3.0f;

                // Jaro-Winkler prefix bonus
                int prefix_len = 0;
                for (int k = 0; k < min(4, min(len1, len2)); k++) {
                    if (s1[k] == s2[k]) prefix_len++;
                    else break;
                }

                float jaro_winkler = jaro + prefix_len * prefix_weight * (1.0f - jaro);
                result[i * n2 + j] = jaro_winkler;

                delete[] s1_matches;
                delete[] s2_matches;
            }
            ''', 'jaro_winkler_kernel')

            # Launch kernel
            block_size = (16, 16)
            grid_size = ((len(names1) + block_size[0] - 1) // block_size[0],
                        (len(names2) + block_size[1] - 1) // block_size[1])

            jaro_winkler_kernel(grid_size, block_size,
                              (names1_gpu, names2_gpu, result_gpu,
                               len(names1), len(names2), max_len1, max_len2, prefix_weight))

            # Transfer result back to CPU
            return cp.asnumpy(result_gpu)

        except Exception as e:
            logger.error(f"CuPy Jaro-Winkler failed: {e}")
            raise

    def _batch_jaro_winkler_torch(self, names1: List[str], names2: List[str],
                                  prefix_weight: float) -> np.ndarray:
        """PyTorch implementation of batch Jaro-Winkler."""
        try:
            import torch
            import torch.nn.functional as F

            # Convert strings to tensors
            max_len = max(max(len(name) for name in names1) if names1 else 0,
                         max(len(name) for name in names2) if names2 else 0)

            # Pad and encode strings
            def encode_names(names, max_length):
                encoded = torch.zeros(len(names), max_length, dtype=torch.long)
                lengths = torch.zeros(len(names), dtype=torch.long)
                for i, name in enumerate(names):
                    name_tensor = torch.tensor([ord(c) for c in name[:max_length]], dtype=torch.long)
                    encoded[i, :len(name_tensor)] = name_tensor
                    lengths[i] = len(name)
                return encoded, lengths

            names1_tensor, lengths1 = encode_names(names1, max_len)
            names2_tensor, lengths2 = encode_names(names2, max_len)

            # Move to GPU
            names1_tensor = names1_tensor.to(self.device)
            names2_tensor = names2_tensor.to(self.device)
            lengths1 = lengths1.to(self.device)
            lengths2 = lengths2.to(self.device)

            # Vectorized Jaro-Winkler calculation
            n1, n2 = len(names1), len(names2)
            result = torch.zeros(n1, n2, device=self.device)

            # Batch processing to manage memory
            batch_size = min(100, n1)  # Process in batches to avoid memory issues

            for i in range(0, n1, batch_size):
                end_i = min(i + batch_size, n1)
                batch_names1 = names1_tensor[i:end_i]
                batch_lengths1 = lengths1[i:end_i]

                # Expand for broadcasting
                batch_names1_exp = batch_names1.unsqueeze(1).expand(-1, n2, -1)  # [batch, n2, max_len]
                names2_exp = names2_tensor.unsqueeze(0).expand(end_i - i, -1, -1)  # [batch, n2, max_len]

                # Character-level matching
                char_matches = (batch_names1_exp == names2_exp).float()  # [batch, n2, max_len]

                # Simple similarity approximation (not full Jaro-Winkler but fast)
                batch_lengths1_exp = batch_lengths1.unsqueeze(1).expand(-1, n2)
                lengths2_exp = lengths2.unsqueeze(0).expand(end_i - i, -1)

                # Count matching characters
                matches = char_matches.sum(dim=2)  # [batch, n2]

                # Normalize by average length
                avg_lengths = (batch_lengths1_exp + lengths2_exp).float() / 2.0
                similarities = matches / torch.clamp(avg_lengths, min=1.0)

                result[i:end_i] = similarities

            return result.cpu().numpy()

        except Exception as e:
            logger.error(f"PyTorch Jaro-Winkler failed: {e}")
            raise

    def _batch_jaro_winkler_numba(self, names1: List[str], names2: List[str],
                                  prefix_weight: float) -> np.ndarray:
        """Numba CUDA implementation of batch Jaro-Winkler."""
        try:
            from numba import cuda
            import math

            # Convert strings to numeric arrays
            max_len1 = max(len(name) for name in names1) if names1 else 0
            max_len2 = max(len(name) for name in names2) if names2 else 0

            names1_array = np.zeros((len(names1), max_len1), dtype=np.int32)
            names2_array = np.zeros((len(names2), max_len2), dtype=np.int32)
            lengths1 = np.zeros(len(names1), dtype=np.int32)
            lengths2 = np.zeros(len(names2), dtype=np.int32)

            for i, name in enumerate(names1):
                for j, char in enumerate(name[:max_len1]):
                    names1_array[i, j] = ord(char)
                lengths1[i] = len(name)

            for i, name in enumerate(names2):
                for j, char in enumerate(name[:max_len2]):
                    names2_array[i, j] = ord(char)
                lengths2[i] = len(name)

            # CUDA kernel for Jaro-Winkler
            @cuda.jit
            def jaro_winkler_kernel(names1, names2, lengths1, lengths2, result, prefix_weight):
                i, j = cuda.grid(2)

                if i >= names1.shape[0] or j >= names2.shape[0]:
                    return

                len1 = lengths1[i]
                len2 = lengths2[j]

                if len1 == 0 and len2 == 0:
                    result[i, j] = 1.0
                    return
                if len1 == 0 or len2 == 0:
                    result[i, j] = 0.0
                    return

                # Simplified Jaro calculation
                match_window = max(len1, len2) // 2 - 1
                if match_window < 0:
                    match_window = 0

                matches = 0
                transpositions = 0

                # Count matches (simplified)
                for k in range(len1):
                    start = max(0, k - match_window)
                    end = min(k + match_window + 1, len2)

                    for l in range(start, end):
                        if names1[i, k] == names2[j, l]:
                            matches += 1
                            break

                if matches == 0:
                    result[i, j] = 0.0
                    return

                # Jaro similarity
                jaro = (matches / len1 + matches / len2 + (matches - transpositions/2) / matches) / 3.0

                # Prefix bonus
                prefix_len = 0
                for k in range(min(4, min(len1, len2))):
                    if names1[i, k] == names2[j, k]:
                        prefix_len += 1
                    else:
                        break

                jaro_winkler = jaro + prefix_len * prefix_weight * (1.0 - jaro)
                result[i, j] = jaro_winkler

            # Allocate GPU memory
            names1_gpu = cuda.to_device(names1_array)
            names2_gpu = cuda.to_device(names2_array)
            lengths1_gpu = cuda.to_device(lengths1)
            lengths2_gpu = cuda.to_device(lengths2)
            result_gpu = cuda.device_array((len(names1), len(names2)), dtype=np.float32)

            # Launch kernel
            threads_per_block = (16, 16)
            blocks_per_grid_x = math.ceil(len(names1) / threads_per_block[0])
            blocks_per_grid_y = math.ceil(len(names2) / threads_per_block[1])
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

            jaro_winkler_kernel[blocks_per_grid, threads_per_block](
                names1_gpu, names2_gpu, lengths1_gpu, lengths2_gpu, result_gpu, prefix_weight
            )

            # Copy result back to host
            return result_gpu.copy_to_host()

        except Exception as e:
            logger.error(f"Numba CUDA Jaro-Winkler failed: {e}")
            raise

class GPUNameMatcher:
    """High-level GPU-accelerated name matcher."""

    def __init__(self, enable_gpu: bool = True, framework: str = None):
        """
        Initialize GPU name matcher.

        Args:
            enable_gpu: Whether to enable GPU acceleration
            framework: Preferred GPU framework
        """
        self.enable_gpu = enable_gpu and gpu_framework.has_gpu
        self.gpu_matcher = GPUStringMatcher(framework) if self.enable_gpu else None

        logger.info(f"GPU acceleration {'enabled' if self.enable_gpu else 'disabled'}")
        if self.enable_gpu:
            logger.info(f"Using framework: {self.gpu_matcher.framework}")

    def batch_similarity_matrix(self, names1: List[str], names2: List[str],
                               algorithm: str = 'jaro_winkler') -> np.ndarray:
        """
        Calculate similarity matrix for two lists of names.

        Args:
            names1: First list of names
            names2: Second list of names
            algorithm: Similarity algorithm ('jaro_winkler', 'levenshtein')

        Returns:
            2D numpy array of similarity scores
        """
        if not self.enable_gpu or not names1 or not names2:
            return self._cpu_similarity_matrix(names1, names2, algorithm)

        try:
            start_time = time.time()

            if algorithm == 'jaro_winkler':
                result = self.gpu_matcher.batch_jaro_winkler_gpu(names1, names2)
            else:
                # Fallback to CPU for unsupported algorithms
                result = self._cpu_similarity_matrix(names1, names2, algorithm)

            gpu_time = time.time() - start_time
            total_comparisons = len(names1) * len(names2)

            logger.info(f"GPU {algorithm}: {total_comparisons:,} comparisons in {gpu_time:.4f}s "
                       f"({total_comparisons/gpu_time:.0f} comp/sec)")

            return result

        except Exception as e:
            logger.warning(f"GPU similarity matrix failed: {e}, falling back to CPU")
            return self._cpu_similarity_matrix(names1, names2, algorithm)

    def _cpu_similarity_matrix(self, names1: List[str], names2: List[str],
                              algorithm: str) -> np.ndarray:
        """CPU fallback for similarity matrix calculation."""
        try:
            from .matcher import jaro_winkler_similarity, damerau_levenshtein_similarity
        except ImportError:
            from matcher import jaro_winkler_similarity, damerau_levenshtein_similarity

        if algorithm == 'jaro_winkler':
            sim_func = jaro_winkler_similarity
        elif algorithm == 'levenshtein':
            sim_func = damerau_levenshtein_similarity
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        result = np.zeros((len(names1), len(names2)), dtype=np.float32)

        for i, name1 in enumerate(names1):
            for j, name2 in enumerate(names2):
                result[i, j] = sim_func(name1, name2)

        return result

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU acceleration information."""
        info = gpu_framework.get_framework_info()
        info['gpu_enabled'] = self.enable_gpu
        if self.enable_gpu and self.gpu_matcher:
            info['active_framework'] = self.gpu_matcher.framework
            info['device_id'] = self.gpu_matcher.device_id
        return info

# Configuration management
class GPUConfig:
    """GPU acceleration configuration."""

    def __init__(self):
        self.enabled = True
        self.framework = None  # Auto-select
        self.device_id = 0
        self.batch_size = 1000
        self.memory_limit_gb = 4.0
        self.fallback_threshold = 10000  # Use CPU for small datasets

    def from_dict(self, config: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.enabled = config.get('enabled', self.enabled)
        self.framework = config.get('framework', self.framework)
        self.device_id = config.get('device_id', self.device_id)
        self.batch_size = config.get('batch_size', self.batch_size)
        self.memory_limit_gb = config.get('memory_limit_gb', self.memory_limit_gb)
        self.fallback_threshold = config.get('fallback_threshold', self.fallback_threshold)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'framework': self.framework,
            'device_id': self.device_id,
            'batch_size': self.batch_size,
            'memory_limit_gb': self.memory_limit_gb,
            'fallback_threshold': self.fallback_threshold
        }

# Global GPU configuration
gpu_config = GPUConfig()

def configure_gpu(config: Dict[str, Any]):
    """Configure GPU acceleration settings."""
    global gpu_config
    gpu_config.from_dict(config)
    logger.info(f"GPU configuration updated: {gpu_config.to_dict()}")

def get_gpu_status() -> Dict[str, Any]:
    """Get comprehensive GPU status information."""
    return {
        'framework_info': gpu_framework.get_framework_info(),
        'config': gpu_config.to_dict(),
        'available': gpu_framework.has_gpu,
        'enabled': gpu_config.enabled
    }

# Convenience functions
def create_gpu_matcher(enable_gpu: bool = None, framework: str = None) -> GPUNameMatcher:
    """Create a GPU name matcher with current configuration."""
    if enable_gpu is None:
        enable_gpu = gpu_config.enabled
    if framework is None:
        framework = gpu_config.framework

    return GPUNameMatcher(enable_gpu=enable_gpu, framework=framework)

# Export main classes and functions
__all__ = [
    'GPUNameMatcher', 'GPUStringMatcher', 'GPUFramework', 'GPUConfig',
    'gpu_framework', 'gpu_config', 'configure_gpu', 'get_gpu_status',
    'create_gpu_matcher', 'gpu_fallback'
]
