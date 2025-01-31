# TensorFlow GPU Setup with CUDA 12.8

## Prerequisites
1. **Check NVIDIA Driver**
```bash
nvidia-smi
```
Should show driver version 525.x or higher

2. **Check CUDA Installation**
```bash
nvcc --version
```
Should show CUDA 12.x

## Installation Steps

1. **System Environment Variables**
Add these to Windows System Environment Variables (System Properties > Advanced > Environment Variables):
```
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
PATH=%CUDA_PATH%\bin;%PATH%
PATH=%CUDA_PATH%\libnvvp;%PATH%
```

2. **Install Visual Studio Components**
- Download Visual Studio 2019 or 2022 Community
- Install "Desktop development with C++"
- Install MSVC v143 build tools

3. **Install CUDA Toolkit**
- Download CUDA 12.8 from NVIDIA website
- Custom installation, select:
  * CUDA
  * CUDA DNN
  * CUDA Development Tools

4. **Install Python Packages**
```bash
# Create fresh environment
python -m venv tf-gpu-env
.\tf-gpu-env\Scripts\activate

# Install packages
pip install tensorflow==2.15.0
pip install nvidia-cudnn-cu12
```

5. **Verify Installation**
Run the verification script:
```bash
python install_tf_gpu.py
```

## Troubleshooting

1. **If GPU Still Not Detected**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices())
   print(tf.test.is_built_with_cuda())
   ```

2. **Check System Path**
   ```bash
   where nvcc
   where cudnn*.dll
   ```

3. **Common Fixes**
   - Restart computer after installation
   - Update NVIDIA drivers
   - Reinstall Visual C++ Redistributable
   - Clear pip cache: `pip cache purge`

4. **Version Compatibility**
   - TensorFlow 2.15.0 → CUDA 12.x
   - NVIDIA Driver ≥ 525.x
   - cuDNN ≥ 8.9.x
