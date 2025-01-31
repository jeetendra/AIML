import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

def setup_tensorflow_gpu():
    # Uninstall existing TensorFlow
    run_command("pip uninstall -y tensorflow tensorflow-gpu")
    
    # Install CUDA-compatible TensorFlow
    run_command("pip install tensorflow==2.15.0")
    run_command("pip install nvidia-cudnn-cu12")
    
    # Verify CUDA installation
    print("\nVerifying CUDA installation...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Print environment variables
    print("\nCUDA Environment Variables:")
    cuda_vars = ['CUDA_PATH', 'PATH', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

if __name__ == "__main__":
    setup_tensorflow_gpu()
