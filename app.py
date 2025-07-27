import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
    input_data[feat] = st.slider(f"{feat} (1=Low, 5=High)", min_value=1, max_value=5, value=3, step=1)

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=features)

if st.button("Predict"):
    pred = clf.predict(input_scaled_df)
    st.success(f"Predicted Group: {pred[0]}") 