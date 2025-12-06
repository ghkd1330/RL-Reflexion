#!/usr/bin/env python3
"""
Environment check script for RL-Project
Verifies CUDA, PyTorch GPU support, and system configuration.
"""

import sys
import platform

def check_python():
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Version Info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()

def check_cuda():
    print("=" * 60)
    print("CUDA and PyTorch Check")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
                
            # Test tensor creation on GPU
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"\n✓ Successfully created and multiplied tensors on GPU")
        else:
            print("⚠ WARNING: CUDA is not available. GPU acceleration will not work.")
            print("  Please ensure NVIDIA drivers and CUDA are properly installed.")
    except ImportError:
        print("✗ PyTorch is not installed. Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error during CUDA check: {e}")
    print()

def check_dependencies():
    print("=" * 60)
    print("Key Dependencies Check")
    print("=" * 60)
    
    dependencies = {
        'ai2thor': 'AI2-THOR Simulator',
        'allennlp': 'AllenNLP',
        'stable_baselines3': 'Stable-Baselines3',
        'transformers': 'Hugging Face Transformers',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
    }
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
    print()

def check_system():
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print()

def main():
    print("\n" + "=" * 60)
    print("RL-Project Environment Check")
    print("=" * 60)
    print()
    
    check_python()
    check_system()
    check_cuda()
    check_dependencies()
    
    print("=" * 60)
    print("Environment check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
