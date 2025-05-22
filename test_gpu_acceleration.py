#!/usr/bin/env python3
"""
GPU Acceleration Testing and Benchmarking Script

This script tests GPU acceleration functionality and provides performance benchmarks
comparing CPU vs GPU implementations for name matching operations.
"""

import time
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.gpu_acceleration import (
        GPUNameMatcher, GPUFramework, get_gpu_status, 
        configure_gpu, create_gpu_matcher
    )
    from src.name_matcher import NameMatcher
    from src.config import get_gpu_config
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"GPU acceleration not available: {e}")
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUBenchmark:
    """Comprehensive GPU acceleration benchmark suite."""
    
    def __init__(self):
        self.results = {}
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict[str, List[str]]:
        """Generate realistic Filipino name test data."""
        
        # Common Filipino names for realistic testing
        first_names = [
            'Juan', 'Maria', 'Jose', 'Ana', 'Carlos', 'Rosa', 'Antonio', 'Carmen',
            'Francisco', 'Luz', 'Pedro', 'Esperanza', 'Manuel', 'Josefa', 'Luis',
            'Remedios', 'Miguel', 'Concepcion', 'Angel', 'Teresita', 'Roberto',
            'Gloria', 'Ricardo', 'Corazon', 'Eduardo', 'Milagros', 'Ramon',
            'Soledad', 'Alfredo', 'Pilar', 'Fernando', 'Natividad', 'Arturo',
            'Rosario', 'Ernesto', 'Dolores', 'Armando', 'Cristina', 'Rodolfo',
            'Leonor', 'Reynaldo', 'Lourdes', 'Rogelio', 'Felicidad', 'Danilo'
        ]
        
        last_names = [
            'Santos', 'Cruz', 'Garcia', 'Lopez', 'Reyes', 'Ramos', 'Mendoza',
            'Torres', 'Flores', 'Gonzales', 'Bautista', 'Villanueva', 'Aquino',
            'Castillo', 'Rivera', 'Morales', 'Dela Cruz', 'Hernandez', 'Perez',
            'Manalo', 'Aguilar', 'Valdez', 'Salazar', 'Pascual', 'Santiago',
            'Soriano', 'Mercado', 'Diaz', 'Gutierrez', 'Fernandez', 'Domingo',
            'Castro', 'Vargas', 'Rosales', 'Marquez', 'Romero', 'Navarro',
            'Tolentino', 'Velasco', 'Cabrera', 'Espiritu', 'Magno', 'Alvarez'
        ]
        
        middle_names = [
            'dela', 'de la', 'delos', 'de los', 'san', 'santa', 'del', 'de'
        ]
        
        # Generate test datasets of different sizes
        datasets = {}
        
        for size in [100, 500, 1000, 2000]:
            names1 = []
            names2 = []
            
            for i in range(size):
                # Generate first dataset
                first = np.random.choice(first_names)
                middle = np.random.choice(middle_names) if np.random.random() > 0.3 else ''
                last = np.random.choice(last_names)
                
                if middle:
                    full_name1 = f"{first} {middle} {last}"
                else:
                    full_name1 = f"{first} {last}"
                names1.append(full_name1)
                
                # Generate second dataset with some variations
                if np.random.random() > 0.7:  # 30% exact matches
                    names2.append(full_name1)
                else:
                    # Add variations
                    first2 = first if np.random.random() > 0.2 else np.random.choice(first_names)
                    middle2 = middle if np.random.random() > 0.3 else np.random.choice(middle_names)
                    last2 = last if np.random.random() > 0.1 else np.random.choice(last_names)
                    
                    if middle2:
                        full_name2 = f"{first2} {middle2} {last2}"
                    else:
                        full_name2 = f"{first2} {last2}"
                    names2.append(full_name2)
            
            datasets[f"size_{size}"] = {
                'names1': names1,
                'names2': names2
            }
        
        return datasets
    
    def test_gpu_detection(self) -> Dict[str, Any]:
        """Test GPU framework detection and availability."""
        logger.info("=== GPU Detection Test ===")
        
        if not GPU_AVAILABLE:
            return {'status': 'failed', 'error': 'GPU module not available'}
        
        try:
            # Test framework detection
            framework = GPUFramework()
            status = get_gpu_status()
            
            result = {
                'status': 'success',
                'frameworks_detected': status['framework_info'],
                'gpu_available': status['available'],
                'preferred_framework': framework.preferred_framework,
                'device_count': framework.gpu_device_count
            }
            
            logger.info(f"GPU Detection Results: {result}")
            return result
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_gpu_configuration(self) -> Dict[str, Any]:
        """Test GPU configuration loading and management."""
        logger.info("=== GPU Configuration Test ===")
        
        if not GPU_AVAILABLE:
            return {'status': 'failed', 'error': 'GPU module not available'}
        
        try:
            # Test configuration loading
            config = get_gpu_config()
            
            # Test configuration modification
            test_config = {
                'enabled': True,
                'framework': 'auto',
                'device_id': 0,
                'batch_size': 500,
                'memory_limit_gb': 2.0,
                'fallback_threshold': 5000
            }
            
            configure_gpu(test_config)
            
            result = {
                'status': 'success',
                'default_config': config,
                'test_config_applied': test_config
            }
            
            logger.info(f"GPU Configuration Test: {result}")
            return result
            
        except Exception as e:
            logger.error(f"GPU configuration test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def benchmark_similarity_algorithms(self) -> Dict[str, Any]:
        """Benchmark GPU vs CPU similarity algorithms."""
        logger.info("=== Similarity Algorithm Benchmark ===")
        
        results = {}
        
        for dataset_name, data in self.test_data.items():
            logger.info(f"Benchmarking {dataset_name}...")
            
            names1 = data['names1']
            names2 = data['names2']
            total_comparisons = len(names1) * len(names2)
            
            dataset_results = {
                'dataset_size': len(names1),
                'total_comparisons': total_comparisons,
                'algorithms': {}
            }
            
            # Test different algorithms
            algorithms = ['jaro_winkler']
            if GPU_AVAILABLE:
                algorithms.append('levenshtein')
            
            for algorithm in algorithms:
                logger.info(f"  Testing {algorithm}...")
                
                # CPU benchmark
                cpu_time, cpu_result = self._benchmark_cpu_similarity(
                    names1, names2, algorithm
                )
                
                # GPU benchmark (if available)
                if GPU_AVAILABLE:
                    gpu_time, gpu_result = self._benchmark_gpu_similarity(
                        names1, names2, algorithm
                    )
                    
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    
                    dataset_results['algorithms'][algorithm] = {
                        'cpu_time': cpu_time,
                        'gpu_time': gpu_time,
                        'speedup': speedup,
                        'cpu_throughput': total_comparisons / cpu_time if cpu_time > 0 else 0,
                        'gpu_throughput': total_comparisons / gpu_time if gpu_time > 0 else 0,
                        'results_match': np.allclose(cpu_result, gpu_result, atol=1e-3) if gpu_result is not None else False
                    }
                else:
                    dataset_results['algorithms'][algorithm] = {
                        'cpu_time': cpu_time,
                        'gpu_time': None,
                        'speedup': None,
                        'cpu_throughput': total_comparisons / cpu_time if cpu_time > 0 else 0,
                        'gpu_throughput': None,
                        'results_match': None
                    }
            
            results[dataset_name] = dataset_results
        
        return results
    
    def _benchmark_cpu_similarity(self, names1: List[str], names2: List[str], 
                                 algorithm: str) -> tuple:
        """Benchmark CPU similarity calculation."""
        try:
            from src.matcher import jaro_winkler_similarity, damerau_levenshtein_similarity
            
            if algorithm == 'jaro_winkler':
                sim_func = jaro_winkler_similarity
            elif algorithm == 'levenshtein':
                sim_func = damerau_levenshtein_similarity
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Warm up
            for i in range(min(10, len(names1))):
                for j in range(min(10, len(names2))):
                    sim_func(names1[i], names2[j])
            
            # Benchmark
            start_time = time.time()
            result = np.zeros((len(names1), len(names2)), dtype=np.float32)
            
            for i, name1 in enumerate(names1):
                for j, name2 in enumerate(names2):
                    result[i, j] = sim_func(name1, name2)
            
            cpu_time = time.time() - start_time
            return cpu_time, result
            
        except Exception as e:
            logger.error(f"CPU benchmark failed: {e}")
            return float('inf'), None
    
    def _benchmark_gpu_similarity(self, names1: List[str], names2: List[str], 
                                 algorithm: str) -> tuple:
        """Benchmark GPU similarity calculation."""
        if not GPU_AVAILABLE:
            return float('inf'), None
        
        try:
            gpu_matcher = create_gpu_matcher(enable_gpu=True)
            
            if not gpu_matcher.enable_gpu:
                logger.warning("GPU not available for benchmark")
                return float('inf'), None
            
            # Warm up
            small_names1 = names1[:min(10, len(names1))]
            small_names2 = names2[:min(10, len(names2))]
            gpu_matcher.batch_similarity_matrix(small_names1, small_names2, algorithm)
            
            # Benchmark
            start_time = time.time()
            result = gpu_matcher.batch_similarity_matrix(names1, names2, algorithm)
            gpu_time = time.time() - start_time
            
            return gpu_time, result
            
        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            return float('inf'), None
    
    def test_dataframe_matching(self) -> Dict[str, Any]:
        """Test GPU-accelerated DataFrame matching."""
        logger.info("=== DataFrame Matching Test ===")
        
        if not GPU_AVAILABLE:
            return {'status': 'failed', 'error': 'GPU module not available'}
        
        try:
            # Create test DataFrames
            data = self.test_data['size_500']
            
            df1 = pd.DataFrame({
                'hh_id': range(len(data['names1'])),
                'first_name': [name.split()[0] for name in data['names1']],
                'middle_name_last_name': [' '.join(name.split()[1:]) for name in data['names1']],
                'birthdate': ['1990-01-01'] * len(data['names1']),
                'province_name': ['Manila'] * len(data['names1'])
            })
            
            df2 = pd.DataFrame({
                'hh_id': range(1000, 1000 + len(data['names2'])),
                'first_name': [name.split()[0] for name in data['names2']],
                'middle_name_last_name': [' '.join(name.split()[1:]) for name in data['names2']],
                'birthdate': ['1990-01-01'] * len(data['names2']),
                'province_name': ['Manila'] * len(data['names2'])
            })
            
            # Test CPU matching
            cpu_matcher = NameMatcher(enable_gpu=False)
            start_time = time.time()
            cpu_results = cpu_matcher.match_dataframes(df1.head(50), df2.head(50))
            cpu_time = time.time() - start_time
            
            # Test GPU matching
            gpu_matcher = NameMatcher(enable_gpu=True)
            start_time = time.time()
            gpu_results = gpu_matcher.match_dataframes_gpu(df1.head(50), df2.head(50))
            gpu_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': cpu_time / gpu_time if gpu_time > 0 else 0,
                'cpu_matches': len(cpu_results),
                'gpu_matches': len(gpu_results),
                'dataframes_tested': True
            }
            
            logger.info(f"DataFrame Matching Test: {result}")
            return result
            
        except Exception as e:
            logger.error(f"DataFrame matching test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete GPU acceleration benchmark suite."""
        logger.info("=== Starting Full GPU Benchmark Suite ===")
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_detection': self.test_gpu_detection(),
            'gpu_configuration': self.test_gpu_configuration(),
            'similarity_benchmarks': self.benchmark_similarity_algorithms(),
            'dataframe_matching': self.test_dataframe_matching()
        }
        
        # Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        logger.info("=== Benchmark Suite Complete ===")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'gpu_available': results['gpu_detection'].get('gpu_available', False),
            'tests_passed': 0,
            'tests_failed': 0,
            'best_speedup': 0,
            'average_speedup': 0,
            'recommendations': []
        }
        
        # Count test results
        for test_name, test_result in results.items():
            if test_name == 'summary':
                continue
            
            if isinstance(test_result, dict):
                if test_result.get('status') == 'success':
                    summary['tests_passed'] += 1
                elif test_result.get('status') == 'failed':
                    summary['tests_failed'] += 1
        
        # Analyze speedups
        speedups = []
        if 'similarity_benchmarks' in results:
            for dataset_name, dataset_results in results['similarity_benchmarks'].items():
                for algorithm, algo_results in dataset_results.get('algorithms', {}).items():
                    speedup = algo_results.get('speedup')
                    if speedup and speedup > 0:
                        speedups.append(speedup)
        
        if speedups:
            summary['best_speedup'] = max(speedups)
            summary['average_speedup'] = sum(speedups) / len(speedups)
        
        # Generate recommendations
        if summary['gpu_available']:
            if summary['best_speedup'] > 2:
                summary['recommendations'].append("GPU acceleration provides significant speedup - recommended for production")
            elif summary['best_speedup'] > 1:
                summary['recommendations'].append("GPU acceleration provides moderate speedup - consider for large datasets")
            else:
                summary['recommendations'].append("GPU acceleration overhead may not be worth it for current dataset sizes")
        else:
            summary['recommendations'].append("Install GPU libraries (CuPy, PyTorch) for acceleration")
        
        return summary

def print_benchmark_results(results: Dict[str, Any]):
    """Print formatted benchmark results."""
    print("\n" + "="*80)
    print("GPU ACCELERATION BENCHMARK RESULTS")
    print("="*80)
    
    # Summary
    summary = results.get('summary', {})
    print(f"\n📊 SUMMARY")
    print(f"   GPU Available: {'✅' if summary.get('gpu_available') else '❌'}")
    print(f"   Tests Passed: {summary.get('tests_passed', 0)}")
    print(f"   Tests Failed: {summary.get('tests_failed', 0)}")
    print(f"   Best Speedup: {summary.get('best_speedup', 0):.2f}x")
    print(f"   Average Speedup: {summary.get('average_speedup', 0):.2f}x")
    
    # GPU Detection
    gpu_detection = results.get('gpu_detection', {})
    print(f"\n🔍 GPU DETECTION")
    if gpu_detection.get('status') == 'success':
        frameworks = gpu_detection.get('frameworks_detected', {})
        print(f"   CuPy: {'✅' if frameworks.get('cupy_available') else '❌'}")
        print(f"   PyTorch: {'✅' if frameworks.get('torch_available') else '❌'}")
        print(f"   Numba CUDA: {'✅' if frameworks.get('numba_cuda_available') else '❌'}")
        print(f"   Preferred: {frameworks.get('preferred_framework', 'None')}")
        print(f"   GPU Devices: {frameworks.get('gpu_device_count', 0)}")
    else:
        print(f"   ❌ {gpu_detection.get('error', 'Unknown error')}")
    
    # Similarity Benchmarks
    similarity_benchmarks = results.get('similarity_benchmarks', {})
    print(f"\n⚡ SIMILARITY BENCHMARKS")
    
    for dataset_name, dataset_results in similarity_benchmarks.items():
        size = dataset_results.get('dataset_size', 0)
        comparisons = dataset_results.get('total_comparisons', 0)
        print(f"\n   📋 {dataset_name.upper()} ({size} names, {comparisons:,} comparisons)")
        
        for algorithm, algo_results in dataset_results.get('algorithms', {}).items():
            cpu_time = algo_results.get('cpu_time', 0)
            gpu_time = algo_results.get('gpu_time', 0)
            speedup = algo_results.get('speedup', 0)
            cpu_throughput = algo_results.get('cpu_throughput', 0)
            gpu_throughput = algo_results.get('gpu_throughput', 0)
            
            print(f"      {algorithm.upper()}:")
            print(f"        CPU: {cpu_time:.4f}s ({cpu_throughput:.0f} comp/sec)")
            if gpu_time:
                print(f"        GPU: {gpu_time:.4f}s ({gpu_throughput:.0f} comp/sec)")
                print(f"        Speedup: {speedup:.2f}x")
            else:
                print(f"        GPU: Not available")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "="*80)

def main():
    """Main benchmark execution."""
    print("GPU Acceleration Testing and Benchmarking")
    print("=========================================")
    
    # Run benchmarks
    benchmark = GPUBenchmark()
    results = benchmark.run_full_benchmark()
    
    # Print results
    print_benchmark_results(results)
    
    # Save results to file
    import json
    with open('gpu_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: gpu_benchmark_results.json")

if __name__ == "__main__":
    main()
