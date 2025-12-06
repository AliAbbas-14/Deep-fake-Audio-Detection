import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import librosa
import tempfile
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Assignment #3", layout="wide")

st.title("🎯 AI Assignment #3 - Deepfake Audio & Defect Prediction")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
task = st.sidebar.radio("Select Task:", 
                        ["Deepfake Audio Detection", "Software Defect Prediction"])
model_type = st.sidebar.selectbox("Select Model:", 
                                  ["SVM", "Logistic Regression", "Perceptron", "DNN"])

# Simple DNN model class
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

if task == "Deepfake Audio Detection":
    st.header("🔊 Deepfake Audio Detection")
    
    uploaded_file = st.file_uploader("Upload audio file (WAV/MP3)", 
                                     type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        
        # Play audio
        st.audio(temp_path)
        
        # Extract features
        with st.spinner("Extracting features..."):
            try:
                # Load audio
                y, sr = librosa.load(temp_path, sr=16000)
                
                # Pad/trim to 16000 samples
                if len(y) > 16000:
                    y = y[:16000]
                else:
                    y = np.pad(y, (0, 16000 - len(y)))
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                
                st.success(f"✅ Extracted {features.shape[1]} features")
                
                # Show features
                with st.expander("View extracted features"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Feature values:")
                        for i, val in enumerate(features[0]):
                            st.write(f"Feature {i+1}: {val:.4f}")
                    with col2:
                        st.line_chart(features[0])
                
                # Load model and predict
                st.subheader("🤖 Prediction Results")
                
                if model_type == "SVM":
                    model = joblib.load("models/deepfake_svm.pkl")
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]
                    confidence = max(prob)
                
                elif model_type == "Logistic Regression":
                    model = joblib.load("models/deepfake_logreg.pkl")
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]
                    confidence = max(prob)
                
                elif model_type == "Perceptron":
                    model = joblib.load("models/deepfake_perceptron.pkl")
                    pred = model.predict(features)[0]
                    confidence = 0.5
                
                elif model_type == "DNN":
                    model = SimpleDNN()
                    model.load_state_dict(torch.load("models/deepfake_dnn.pt", 
                                                     map_location=torch.device('cpu')))
                    model.eval()
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features)
                        output = model(features_tensor)
                        pred = 1 if output.item() > 0.5 else 0
                        confidence = output.item()
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    if pred == 1:
                        st.error("❌ FAKE Audio")
                    else:
                        st.success("✅ REAL Audio")
                
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                with col3:
                    st.progress(float(confidence))
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

else:  # Software Defect Prediction
    st.header("🐛 Software Defect Prediction")
    
    st.info("Enter 40 comma-separated feature values")
    
    # Sample input
    sample = ", ".join([str(i*0.1) for i in range(40)])
    
    features_text = st.text_area("Features:", value=sample, height=150)
    
    if st.button("Predict", type="primary"):
        try:
            # Parse features
            features = [float(x.strip()) for x in features_text.split(",")]
            
            if len(features) != 40:
                st.error(f"Need 40 features, got {len(features)}")
            else:
                # Preprocess
                scaler = StandardScaler()
                features_array = np.array(features).reshape(1, -1)
                processed = scaler.fit_transform(features_array)
                
                # Load model and predict
                if model_type == "SVM":
                    model = joblib.load("models/defect_svm.pkl")
                    pred = model.predict(processed)[0]
                    prob = model.predict_proba(processed)[0]
                    confidence = max(prob)
                
                elif model_type == "Logistic Regression":
                    model = joblib.load("models/defect_logreg.pkl")
                    pred = model.predict(processed)[0]
                    prob = model.predict_proba(processed)[0]
                    confidence = max(prob)
                
                elif model_type == "Perceptron":
                    model = joblib.load("models/defect_perceptron.pkl")
                    pred = model.predict(processed)[0]
                    confidence = 0.5
                
                elif model_type == "DNN":
                    model = SimpleDNN()
                    model.load_state_dict(torch.load("models/defect_dnn.pt", 
                                                     map_location=torch.device('cpu')))
                    model.eval()
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(processed)
                        output = model(features_tensor)
                        pred = 1 if output.item() > 0.5 else 0
                        confidence = output.item()
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    if pred == 1:
                        st.error("🚨 DEFECT Detected")
                    else:
                        st.success("✅ NO Defect")
                
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                st.progress(float(confidence))
                
        except ValueError:
            st.error("Please enter valid numbers separated by commas")

# Footer
st.markdown("---")
st.caption("AI Assignment #3 - Created with Streamlit")