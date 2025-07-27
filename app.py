import streamlit as st
import pandas as pd
import pickle
import os

# Load model and scaler with error handling
try:
    # Try current directory first
    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'
    
    # If not found, try sAT folder
    if not os.path.exists(model_path):
        model_path = 'sAT/model.pkl'
        scaler_path = 'sAT/scaler.pkl'
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

features = [
    'Energy Level', 'Intensity', 'Exercise Needs', 'Potential For Playfulness',
    'Easy To Train', 'Intelligence', 'Kid-Friendly', 'Dog Friendly'
]

st.title("Dog Breed Group Predictor")
st.write("""
Adjust the sliders below to match your dog's characteristics (1 = lowest, 5 = highest), then click Predict to see the likely breed group.
""")

input_data = {}
for feat in features:
    input_data[feat] = st.slider(f"{feat} (1=Low, 5=High)", 1, 5, 3)

if st.button("Predict Dog Breed Group"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=features)
    
    predicted_group = clf.predict(input_scaled_df)
    st.success(f"Predicted Dog Breed Group: **{predicted_group[0]}**") 