import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Generate data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Scale data to [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# Use a ZZFeatureMap for encoding

feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
# Classical SVM with quantum kernel
qsvc = SVC(kernel=quantum_kernel.evaluate)
qsvc.fit(X_train, y_train)

# Test accuracy
accuracy = qsvc.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
