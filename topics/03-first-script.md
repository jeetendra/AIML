# Creating and Running Your First Python Script

## Project Structure
First, create this folder structure:
```
AIML/
├── scripts/
│   ├── hello.py
│   └── test_env.py
├── notebooks/
│   └── first_notebook.ipynb
└── data/
    └── README.md
```

## Creating Scripts

### 1. Basic Script
Create your first script at `scripts/hello.py`:
```python
# Simple test script
print("Hello AI/ML World!")

# Test numpy
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}")
```

### 2. Environment Test Script
Create `scripts/test_env.py`:
```python
# Test all major dependencies
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Create some test data
data = np.random.randn(100)
df = pd.DataFrame({'values': data})

# Create a simple plot
plt.figure(figsize=(8, 4))
plt.hist(data, bins=20)
plt.title('Test Histogram')
plt.savefig('test_plot.png')
plt.close()

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Running Scripts

### Method 1: Command Line
```bash
# Activate your environment first
# Windows
.\aiml-env\Scripts\activate
# Unix/macOS
source aiml-env/bin/activate

# Run scripts
python scripts/hello.py
python scripts/test_env.py
```

### Method 2: VS Code
1. Open VS Code
2. Open your AIML folder
3. Select your Python interpreter (bottom left corner)
4. Right-click script -> "Run Python File in Terminal"
   OR
5. Click the ▶️ (play) button in top right

### Method 3: Jupyter Notebook
Create `notebooks/first_notebook.ipynb`:
```python
# Cell 1
print("Hello from Jupyter!")

# Cell 2
import numpy as np
data = np.random.randn(1000)
print(f"Mean: {data.mean():.2f}")
```

Run with:
```bash
jupyter notebook notebooks/first_notebook.ipynb
```

## Common Issues & Solutions

1. **Module Not Found**
   ```bash
   # Check if you're in the right environment
   which python  # Unix/macOS
   where python  # Windows
   
   # Install missing module
   pip install missing_module
   ```

2. **Wrong Working Directory**
   ```python
   # Add to start of script
   import os
   print("Working directory:", os.getcwd())
   ```

3. **CUDA/GPU Issues**
   ```python
   # Add to your script
   import torch
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))
   else:
       print("No GPU available, using CPU")
   ```

## Best Practices
1. Always use virtual environment
2. Start script names with clear purpose (e.g., `train_model.py`)
3. Add comments and docstrings
4. Use proper error handling:
```python
try:
    import torch
except ImportError:
    print("Please install PyTorch: pip install torch")
    exit(1)
```

## Next Steps
- Try modifying the sample scripts
- Create a script that loads and processes data
- Experiment with different ML libraries
- Learn about debugging in VS Code

[← Back to Development Environment Setup](./02-dev-environment-setup.md)
