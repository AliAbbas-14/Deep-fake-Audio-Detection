import joblib
import torch
import torch.nn as nn
import numpy as np
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("Generating dummy models for testing...")

# Generate more realistic dummy data
np.random.seed(42)
X_dummy = np.random.randn(100, 40)  # 40 features (MFCCs)
y_dummy = np.random.randint(0, 2, 100)

print(f"Dummy data shape: X={X_dummy.shape}, y={y_dummy.shape}")

# =============== SVM Models ===============
print("\nTraining SVM models...")
from sklearn.svm import SVC

# Deepfake SVM
deepfake_svm = SVC(probability=True, random_state=42)
deepfake_svm.fit(X_dummy, y_dummy)
joblib.dump(deepfake_svm, "models/deepfake_svm.pkl")
print("✓ Deepfake SVM saved")

# Defect SVM
defect_svm = SVC(probability=True, random_state=42)
defect_svm.fit(X_dummy, y_dummy)
joblib.dump(defect_svm, "models/defect_svm.pkl")
print("✓ Defect SVM saved")

# =============== Logistic Regression Models ===============
print("\nTraining Logistic Regression models...")
from sklearn.linear_model import LogisticRegression

# Deepfake Logistic Regression
deepfake_logreg = LogisticRegression(random_state=42, max_iter=1000)
deepfake_logreg.fit(X_dummy, y_dummy)
joblib.dump(deepfake_logreg, "models/deepfake_logreg.pkl")
print("✓ Deepfake Logistic Regression saved")

# Defect Logistic Regression
defect_logreg = LogisticRegression(random_state=42, max_iter=1000)
defect_logreg.fit(X_dummy, y_dummy)
joblib.dump(defect_logreg, "models/defect_logreg.pkl")
print("✓ Defect Logistic Regression saved")

# =============== Perceptron Models ===============
print("\nTraining Perceptron models...")
from sklearn.linear_model import Perceptron

# Deepfake Perceptron
deepfake_perceptron = Perceptron(random_state=42, max_iter=1000)
deepfake_perceptron.fit(X_dummy, y_dummy)
joblib.dump(deepfake_perceptron, "models/deepfake_perceptron.pkl")
print("✓ Deepfake Perceptron saved")

# Defect Perceptron
defect_perceptron = Perceptron(random_state=42, max_iter=1000)
defect_perceptron.fit(X_dummy, y_dummy)
joblib.dump(defect_perceptron, "models/defect_perceptron.pkl")
print("✓ Defect Perceptron saved")

# =============== DNN Models ===============
print("\nTraining DNN models...")

class SimpleDNN(nn.Module):
    def __init__(self, input_size=40):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Deepfake DNN
deepfake_dnn = SimpleDNN(input_size=40)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(deepfake_dnn.parameters(), lr=0.001)

# Convert numpy arrays to torch tensors
X_tensor = torch.FloatTensor(X_dummy)
y_tensor = torch.FloatTensor(y_dummy).unsqueeze(1)

# Simple training loop
deepfake_dnn.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = deepfake_dnn(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

deepfake_dnn.eval()
torch.save(deepfake_dnn.state_dict(), "models/deepfake_dnn.pt")
print("✓ Deepfake DNN saved")

# Defect DNN
defect_dnn = SimpleDNN(input_size=40)
optimizer = torch.optim.Adam(defect_dnn.parameters(), lr=0.001)

defect_dnn.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = defect_dnn(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

defect_dnn.eval()
torch.save(defect_dnn.state_dict(), "models/defect_dnn.pt")
print("✓ Defect DNN saved")

# =============== Test the models ===============
print("\n" + "="*50)
print("Testing saved models...")

# Test loading SVM
test_svm = joblib.load("models/deepfake_svm.pkl")
test_pred = test_svm.predict(X_dummy[:5])
print(f"SVM test prediction: {test_pred}")

# Test loading Logistic Regression
test_logreg = joblib.load("models/deepfake_logreg.pkl")
test_pred = test_logreg.predict(X_dummy[:5])
print(f"LogReg test prediction: {test_pred}")

# Test loading DNN
test_dnn = SimpleDNN()
test_dnn.load_state_dict(torch.load("models/deepfake_dnn.pt", map_location=torch.device('cpu')))
test_dnn.eval()
with torch.no_grad():
    test_output = test_dnn(torch.FloatTensor(X_dummy[:5]))
    test_pred = (test_output > 0.5).float()
print(f"DNN test prediction: {test_pred.squeeze().numpy()}")

# =============== Verify file sizes ===============
print("\n" + "="*50)
print("File sizes:")
for file in os.listdir("models"):
    file_path = os.path.join("models", file)
    size = os.path.getsize(file_path)
    print(f"{file}: {size} bytes")

print("\n" + "="*50)
print("✅ All models generated successfully!")

print(f"Models saved in: {os.path.abspath('models')}")
