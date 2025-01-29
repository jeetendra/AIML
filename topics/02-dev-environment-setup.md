# Development Environment Setup for AI/ML

## Basic Setup Requirements
1. **Hardware Requirements**
   - Minimum: 8GB RAM, 4-core CPU
   - Recommended: 16GB+ RAM, 8-core CPU
   - GPU: NVIDIA GPU with 6GB+ VRAM (for deep learning)

2. **Operating System**
   - Windows 11/10 with WSL2
   - macOS 12+
   - Ubuntu 20.04/22.04 LTS

## Step-by-Step Installation Guide

### 1. Python Environment
```bash
# Install Python 3.10+ and tools
python -m pip install --upgrade pip
pip install virtualenv

# Create and activate virtual environment
python -m venv aiml-env
# On Windows
.\aiml-env\Scripts\activate
# On Unix/macOS
source aiml-env/bin/activate
```

### 2. Essential Python Packages
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
pip install torch torchvision torchaudio
pip install tensorflow
pip install transformers
```

### 3. Development Tools
1. **Code Editor/IDE**
   - VSCode with extensions:
     * Python
     * Jupyter
     * Python Test Explorer
     * GitLens
   - PyCharm Professional (alternative)

2. **Version Control**
   ```bash
   # Install Git
   # Windows: https://git-scm.com/download/win
   # macOS
   brew install git
   # Ubuntu
   sudo apt install git
   
   # Configure Git
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### 4. GPU Setup (for Deep Learning)
1. **NVIDIA Drivers**
   - Download latest from: https://www.nvidia.com/Download/index.aspx
   - Install CUDA Toolkit 11.8+
   - Install cuDNN

2. **Verify GPU Setup**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

### 5. Docker Setup (Optional but Recommended)
```bash
# Install Docker Desktop
# Test installation
docker --version
docker run hello-world

# Pull common AI/ML images
docker pull jupyter/scipy-notebook
docker pull tensorflow/tensorflow:latest-gpu
```

## Verification Steps

### 1. Test Python Environment
```python
# test_environment.py
import sys
import numpy
import pandas
import sklearn
import torch
import tensorflow as tf

print(f"Python version: {sys.version}")
print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
```

### 2. Test Jupyter Setup
```bash
jupyter notebook
# Should open browser with Jupyter interface
```

## Common Issues and Solutions

1. **GPU Not Detected**
   - Verify NVIDIA drivers
   - Check CUDA version compatibility
   - Reinstall PyTorch/TensorFlow with GPU support

2. **Package Conflicts**
   ```bash
   # Create new environment
   conda create -n fresh-env python=3.10
   # Or with virtualenv
   python -m venv fresh-env
   ```

3. **Memory Issues**
   ```python
   # Add to your scripts
   import torch
   torch.cuda.empty_cache()
   ```

## Next Steps
- Complete the [Python Basics Tutorial](./03-python-basics.md)
- Try running sample ML models
- Set up your first project repository

[← Back to Introduction](./01-introduction.md)
