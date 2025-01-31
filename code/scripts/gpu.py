import os
import subprocess

def check_system_cuda():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        print("\n=== System CUDA Information ===")
        print(nvcc_output)
        
        cuda_path = os.environ.get('CUDA_PATH', 'Not set')
        path = os.environ.get('PATH', '')
        
        print(f"CUDA_PATH: {cuda_path}")
        print(f"CUDA in PATH: {'CUDA' in path}")
    except Exception as e:
        print(f"Error checking system CUDA: {e}")

def check_pytorch_gpu():
    try:
        import torch
        print("\n=== PyTorch GPU Information ===")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch is not installed")
    except Exception as e:
        print(f"Error checking PyTorch GPU: {e}")

def check_tensorflow_gpu():
    try:
        import tensorflow as tf
        print("\n=== TensorFlow GPU Information ===")
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Set memory growth to prevent TF from taking all GPU memory
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("\nGPU Devices Found:")
            for gpu in gpus:
                print(f"  {gpu}")
            
            # Get GPU device details
            for gpu in gpus:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"\nGPU Details: {gpu_details}")
        else:
            print("\nNo GPU devices found. Running on CPU only.")
            print("\nPossible issues:")
            print("1. TensorFlow not installed with CUDA support")
            print("2. Environment variables not set correctly")
            print("3. Incompatible CUDA version")
            print("\nTry reinstalling TensorFlow with:")
            print("pip install tensorflow==2.15.0")
            print("pip install nvidia-cudnn-cu12")

        # Try to get CUDA info from TensorFlow build
        try:
            from tensorflow.python.platform import build_info as build
            if hasattr(build, 'build_info'):
                cuda_version = build.build_info.get('cuda_version', 'Not available')
                cudnn_version = build.build_info.get('cudnn_version', 'Not available')
                print(f"\nCUDA Version: {cuda_version}")
                print(f"cuDNN Version: {cudnn_version}")
        except Exception:
            print("\nCUDA/cuDNN version information not available")
            
    except ImportError:
        print("TensorFlow is not installed")
    except Exception as e:
        print(f"Error checking TensorFlow GPU: {e}")

if __name__ == "__main__":
    print("Checking GPU Support...")
    check_system_cuda()
    check_pytorch_gpu()
    check_tensorflow_gpu()