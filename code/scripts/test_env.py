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