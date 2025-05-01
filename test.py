import numpy as np
from sklearn.metrics import confusion_matrix

# Generate random binary values (0 or 1) for actual and predicted classifications
np.random.seed(42)  # For reproducibility
ytrue = np.random.randint(0, 2, 100)  # 100 samples
ypred = np.random.randint(0, 2, 100)  # Random predictions

# Compute the confusion matrix
cm = confusion_matrix(ytrue, ypred)

print("Confusion Matrix:")
print(cm)
