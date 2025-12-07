# create_models.py
import joblib
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron

print("Creating model files for AI Assignment #3...")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Generate dummy training data (replace with real data)
np.random.seed(42)
X_dummy = np.random.randn(1000, 40)  # 40 features
y_dummy_audio = np.random.randint(0, 2, 1000)  # Binary for audio
y_dummy_defect = np.random.randint(0, 2, 1000)  # Binary for defect

print(f"Generated dummy data: {X_dummy.shape}")

# ==================== DNN Model Definition ====================
class SimpleDNN(nn.Module):
    def __init__(self, input_size=40):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# ==================== DEEPFAKE AUDIO MODELS ====================
print("\n" + "="*50)
print("Creating Deepfake Audio Detection Models...")

# 1. Deepfake SVM
print("Creating deepfake_svm.pkl...")
deepfake_svm = SVC(probability=True, random_state=42)
deepfake_svm.fit(X_dummy, y_dummy_audio)
joblib.dump(deepfake_svm, "models/deepfake_svm.pkl")
print(f"✓ deepfake_svm.pkl created ({os.path.getsize('models/deepfake_svm.pkl')} bytes)")

# 2. Deepfake Logistic Regression
print("Creating deepfake_logreg.pkl...")
deepfake_logreg = LogisticRegression(random_state=42, max_iter=1000)
deepfake_logreg.fit(X_dummy, y_dummy_audio)
joblib.dump(deepfake_logreg, "models/deepfake_logreg.pkl")
print(f"✓ deepfake_logreg.pkl created ({os.path.getsize('models/deepfake_logreg.pkl')} bytes)")

# 3. Deepfake Perceptron
print("Creating deepfake_perceptron.pkl...")
deepfake_perceptron = Perceptron(random_state=42, max_iter=1000)
deepfake_perceptron.fit(X_dummy, y_dummy_audio)
joblib.dump(deepfake_perceptron, "models/deepfake_perceptron.pkl")
print(f"✓ deepfake_perceptron.pkl created ({os.path.getsize('models/deepfake_perceptron.pkl')} bytes)")

# 4. Deepfake DNN
print("Creating deepfake_dnn.pt...")
deepfake_dnn = SimpleDNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(deepfake_dnn.parameters(), lr=0.001)

# Convert to tensors
X_tensor = torch.FloatTensor(X_dummy)
y_tensor = torch.FloatTensor(y_dummy_audio).unsqueeze(1)

# Train for a few epochs
deepfake_dnn.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = deepfake_dnn(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

deepfake_dnn.eval()
torch.save(deepfake_dnn.state_dict(), "models/deepfake_dnn.pt")
print(f"✓ deepfake_dnn.pt created ({os.path.getsize('models/deepfake_dnn.pt')} bytes)")

# ==================== DEFECT PREDICTION MODELS ====================
print("\n" + "="*50)
print("Creating Software Defect Prediction Models...")

# 1. Defect SVM
print("Creating defect_svm.pkl...")
defect_svm = SVC(probability=True, random_state=42)
defect_svm.fit(X_dummy, y_dummy_defect)
joblib.dump(defect_svm, "models/defect_svm.pkl")
print(f"✓ defect_svm.pkl created ({os.path.getsize('models/defect_svm.pkl')} bytes)")

# 2. Defect Logistic Regression
print("Creating defect_logreg.pkl...")
defect_logreg = LogisticRegression(random_state=42, max_iter=1000)
defect_logreg.fit(X_dummy, y_dummy_defect)
joblib.dump(defect_logreg, "models/defect_logreg.pkl")
print(f"✓ defect_logreg.pkl created ({os.path.getsize('models/defect_logreg.pkl')} bytes)")

# 3. Defect Perceptron
print("Creating defect_perceptron.pkl...")
defect_perceptron = Perceptron(random_state=42, max_iter=1000)
defect_perceptron.fit(X_dummy, y_dummy_defect)
joblib.dump(defect_perceptron, "models/defect_perceptron.pkl")
print(f"✓ defect_perceptron.pkl created ({os.path.getsize('models/defect_perceptron.pkl')} bytes)")

# 4. Defect DNN
print("Creating defect_dnn.pt...")
defect_dnn = SimpleDNN()
optimizer = torch.optim.Adam(defect_dnn.parameters(), lr=0.001)

defect_dnn.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = defect_dnn(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

defect_dnn.eval()
torch.save(defect_dnn.state_dict(), "models/defect_dnn.pt")
print(f"✓ defect_dnn.pt created ({os.path.getsize('models/defect_dnn.pt')} bytes)")

# ==================== VERIFICATION ====================
print("\n" + "="*50)
print("Verifying created models...")

# Test loading SVM
test_svm = joblib.load("models/deepfake_svm.pkl")
test_pred = test_svm.predict(X_dummy[:5])
print(f"✓ SVM test prediction: {test_pred}")

# Test loading DNN
test_dnn = SimpleDNN()
test_dnn.load_state_dict(torch.load("models/deepfake_dnn.pt", map_location='cpu'))
test_dnn.eval()
with torch.no_grad():
    test_output = test_dnn(torch.FloatTensor(X_dummy[:5]))
    test_pred = (test_output > 0.5).float()
print(f"✓ DNN test prediction: {test_pred.squeeze().numpy()}")

print("\n" + "="*50)
print("✅ All 8 model files created successfully!")
print("Models are now ready for Streamlit deployment.")
