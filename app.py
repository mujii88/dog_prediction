import streamlit as st
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for beautiful styling
st.set_page_config(
    page_title="🐕 Dog Breed Predictor",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 0;
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B6B;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    
    .prediction-card h2 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-card p {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        background: rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .slider-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model_path = 'model.pkl'
        scaler_path = 'scaler.pkl'
        
        if not os.path.exists(model_path):
            model_path = 'sAT/model.pkl'
            scaler_path = 'sAT/scaler.pkl'
        
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return clf, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

clf, scaler = load_model()

if clf is None:
    st.error("Failed to load model. Please check if model.pkl and scaler.pkl exist.")
    st.stop()

# Updated features
features_expanded = [
    'Energy Level', 'Intensity', 'Exercise Needs', 'Potential For Playfulness',
    'Easy To Train', 'Intelligence', 'Kid-Friendly', 'Dog Friendly',
    'Affectionate With Family', 'Friendly Toward Strangers', 'Amount Of Shedding',
    'Easy To Groom', 'General Health', 'Potential For Weight Gain', 'Size',
    'Potential For Mouthiness', 'Tendency To Bark Or Howl', 'Wanderlust Potential'
]

# Main header
st.markdown("""
<div class="main-header">
    <h1>🐕 Dog Breed Group Predictor</h1>
    <p>Discover your perfect canine companion based on personality traits!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;">
    <h3>🎯 Quick Guide</h3>
    <p>Adjust the sliders to match your ideal dog's characteristics:</p>
    <ul>
        <li>1 = Very Low</li>
        <li>3 = Moderate</li>
        <li>5 = Very High</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="slider-container">
        <h3>🎮 Dog Characteristics</h3>
        <p>Slide to set your preferred dog traits:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sliders in a more organized way
    input_data = {}
    
    # Group features by category
    energy_features = ['Energy Level', 'Intensity', 'Exercise Needs', 'Potential For Playfulness']
    training_features = ['Easy To Train', 'Intelligence', 'Potential For Mouthiness']
    social_features = ['Kid-Friendly', 'Dog Friendly', 'Affectionate With Family', 'Friendly Toward Strangers']
    care_features = ['Amount Of Shedding', 'Easy To Groom', 'General Health', 'Potential For Weight Gain']
    other_features = ['Size', 'Tendency To Bark Or Howl', 'Wanderlust Potential']
    
    # Energy & Activity
    st.markdown("**⚡ Energy & Activity Level**")
    for feat in energy_features:
        input_data[feat] = st.slider(f"{feat}", 1, 5, 3, help=f"Rate the {feat.lower()} (1=Low, 5=High)")
    
    st.markdown("---")
    
    # Training & Intelligence
    st.markdown("**🎓 Training & Intelligence**")
    for feat in training_features:
        input_data[feat] = st.slider(f"{feat}", 1, 5, 3, help=f"Rate the {feat.lower()} (1=Low, 5=High)")
    
    st.markdown("---")
    
    # Social Behavior
    st.markdown("**🤝 Social Behavior**")
    for feat in social_features:
        input_data[feat] = st.slider(f"{feat}", 1, 5, 3, help=f"Rate the {feat.lower()} (1=Low, 5=High)")
    
    st.markdown("---")
    
    # Care & Maintenance
    st.markdown("**🧹 Care & Maintenance**")
    for feat in care_features:
        input_data[feat] = st.slider(f"{feat}", 1, 5, 3, help=f"Rate the {feat.lower()} (1=Low, 5=High)")
    
    st.markdown("---")
    
    # Other Characteristics
    st.markdown("**📏 Other Characteristics**")
    for feat in other_features:
        input_data[feat] = st.slider(f"{feat}", 1, 5, 3, help=f"Rate the {feat.lower()} (1=Low, 5=High)")

with col2:
    st.markdown("""
    <div class="info-box">
        <h4>💡 Tips</h4>
        <ul>
            <li>Higher energy = More exercise needed</li>
            <li>Higher trainability = Easier to train</li>
            <li>Higher friendliness = Better with people</li>
            <li>Lower shedding = Less grooming needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current selections
    st.markdown("**📊 Your Selections**")
    for feat, value in input_data.items():
        st.markdown(f"**{feat}:** {value}/5")

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Dog Breed Group", use_container_width=True)

# Prediction result
if predict_button:
    try:
        # Create DataFrame with proper column names
        input_df = pd.DataFrame([input_data], columns=features_expanded)
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=features_expanded)
        
        predicted_group = clf.predict(input_scaled_df)[0]
        
        # Display result with beautiful styling
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎉 Prediction Result</h2>
            <p>Your ideal dog belongs to the <strong>{predicted_group}</strong> group!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add some fun facts based on the prediction
        breed_info = {
            "Working Dogs": "Strong, intelligent dogs bred for specific tasks like herding, guarding, or pulling sleds.",
            "Sporting Dogs": "Energetic dogs bred for hunting and retrieving, great for active families.",
            "Hound Dogs": "Dogs with excellent tracking abilities, known for their keen sense of smell.",
            "Terrier Dogs": "Small to medium dogs with big personalities, originally bred for hunting vermin.",
            "Toy Dogs": "Small companion dogs perfect for apartment living and lap cuddles.",
            "Non-Sporting Dogs": "Diverse group of dogs that don't fit into other categories.",
            "Herding Dogs": "Intelligent dogs bred to herd livestock, very trainable and active.",
            "Mixed Breed Dogs": "Unique dogs with diverse backgrounds, often healthier than purebreds."
        }
        
        if predicted_group in breed_info:
            st.markdown(f"""
            <div class="info-box">
                <h4>ℹ️ About {predicted_group}</h4>
                <p>{breed_info[predicted_group]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show feature importance (if using Random Forest)
        if hasattr(clf, 'feature_importances_'):
            st.markdown("**📈 Feature Importance**")
            feature_importance = pd.DataFrame({
                'Feature': features_expanded,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('Feature'))
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🐕 Made with ❤️ for dog lovers everywhere</p>
    <p>Use this tool to find your perfect canine companion!</p>
</div>
""", unsafe_allow_html=True) 