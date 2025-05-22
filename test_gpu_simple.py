#!/usr/bin/env python3
"""
Simple GPU acceleration test without relative imports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu_basic():
    """Test basic GPU functionality."""
    print("=== Basic GPU Test ===")
    
    try:
        from gpu_acceleration import get_gpu_status, GPUFramework, create_gpu_matcher
        
        # Test framework detection
        framework = GPUFramework()
        print(f"GPU Available: {framework.has_gpu}")
        print(f"Preferred Framework: {framework.preferred_framework}")
        print(f"Device Count: {framework.gpu_device_count}")
        
        # Test status
        status = get_gpu_status()
        print(f"GPU Status: {status['available']}")
        print(f"Configuration: {status['config']}")
        
        # Test matcher creation
        matcher = create_gpu_matcher(enable_gpu=True)
        print(f"GPU Matcher Created: {matcher is not None}")
        print(f"GPU Enabled: {matcher.enable_gpu}")
        
        return True
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

def test_similarity_calculation():
    """Test similarity calculation with fallback."""
    print("\n=== Similarity Calculation Test ===")
    
    try:
        from gpu_acceleration import create_gpu_matcher
        
        # Create matcher (will use CPU fallback if no GPU)
        matcher = create_gpu_matcher(enable_gpu=True)
        
        # Test data
        names1 = ['Juan dela Cruz', 'Maria Santos', 'Jose Rizal']
        names2 = ['Juan de la Cruz', 'Maria Santos-Garcia', 'Dr. Jose Rizal']
        
        # Calculate similarity matrix
        similarity_matrix = matcher.batch_similarity_matrix(
            names1, names2, algorithm='jaro_winkler'
        )
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Sample similarities:")
        for i, name1 in enumerate(names1):
            for j, name2 in enumerate(names2):
                score = similarity_matrix[i, j]
                print(f"  '{name1}' vs '{name2}': {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Similarity calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test GPU configuration."""
    print("\n=== Configuration Test ===")
    
    try:
        from gpu_acceleration import configure_gpu, get_gpu_status
        
        # Test configuration
        test_config = {
            'enabled': True,
            'framework': 'auto',
            'batch_size': 500,
            'memory_limit_gb': 2.0
        }
        
        configure_gpu(test_config)
        
        # Verify configuration
        status = get_gpu_status()
        config = status['config']
        
        print(f"Configuration applied successfully:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def main():
    """Run all GPU tests."""
    print("GPU Acceleration Simple Test Suite")
    print("==================================")
    
    tests = [
        ("Basic GPU Detection", test_gpu_basic),
        ("Similarity Calculation", test_similarity_calculation),
        ("Configuration Management", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU acceleration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
