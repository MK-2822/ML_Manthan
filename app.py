import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from datetime import datetime


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fresh-card {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .rotten-card {
        background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model("model1.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
# ==================== PREPROCESSING ====================
def preprocess_image(image, target_size=(240, 240)):
    """
    Preprocess image for model prediction
    NOTE: Model `model1.h5` expects inputs of size 240x240 (training input),
    so default `target_size` is set to (240, 240).
    Adjust this only if you retrain the model with a different input size.
    """
    # Resize image
    img = image.resize(target_size)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize to 0-1 range (adjust if you used different normalization)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ==================== FRESHNESS CALCULATION ====================
def calculate_freshness_percentage(predictions, predicted_class):
    """
    Calculate freshness percentage based on model confidence
    
    Logic:
    - If predicted as "Fresh" with high confidence → High freshness %
    - If predicted as "Rotten" with high confidence → Low freshness %
    - Confidence becomes the freshness indicator
    """
    fresh_confidence = predictions[0][0] * 100  # Assuming index 0 is Fresh
    rotten_confidence = predictions[0][1] * 100  # Assuming index 1 is Rotten
    
    # Calculate freshness percentage
    # If model says Fresh with 95% confidence → 95% fresh
    # If model says Rotten with 90% confidence → 10% fresh
    
    if predicted_class == "Fresh":
        freshness_percentage = fresh_confidence
    else:
        freshness_percentage = 100 - rotten_confidence
    
    return freshness_percentage, fresh_confidence, rotten_confidence

def get_freshness_level(freshness_percentage):
    """Categorize freshness into levels"""
    if freshness_percentage >= 80:
        return "Excellent", "🟢", "#4CAF50"
    elif freshness_percentage >= 60:
        return "Good", "🟡", "#FFC107"
    elif freshness_percentage >= 40:
        return "Fair", "🟠", "#FF9800"
    elif freshness_percentage >= 20:
        return "Poor", "🔴", "#FF5722"
    else:
        return "Spoiled", "⚫", "#F44336"

def get_recommendation(freshness_percentage):
    """Provide recommendations based on freshness"""
    if freshness_percentage >= 80:
        return "✅ Safe to consume. Fruit is fresh and healthy!"
    elif freshness_percentage >= 60:
        return "✅ Good condition. Consume soon for best quality."
    elif freshness_percentage >= 40:
        return "⚠️ Moderate freshness. Consume immediately or use in cooking."
    elif freshness_percentage >= 20:
        return "⚠️ Low freshness. Not recommended for raw consumption."
    else:
        return "❌ Spoiled. Do not consume. Discard immediately."

# ==================== GAUGE CHART ====================
def create_gauge_chart(freshness_percentage):
    """Create a gauge chart for freshness percentage"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = freshness_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Freshness Level", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "lightgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#FFCDD2'},
                {'range': [20, 40], 'color': '#FFE0B2'},
                {'range': [40, 60], 'color': '#FFF9C4'},
                {'range': [60, 80], 'color': '#DCEDC8'},
                {'range': [80, 100], 'color': '#C8E6C9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">🍎 AI Fruit Freshness Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a fruit image to analyze its freshness level using Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/ingredients.png", width=100)
        st.title("About")
        st.info(
            """
            This AI-powered application uses a Convolutional Neural Network (CNN) 
            to classify fruit freshness levels.
            
            **Features:**
            - Real-time freshness detection
            - Confidence scoring
            - Freshness percentage calculation
            - Recommendations
            
            **Supported Fruits:**
            - Apples
            - Bananas
            - Oranges
            - And more!
            """
        )
        
        st.title("How to Use")
        st.markdown("""
        1. 📤 Upload a fruit image
        2. 🔍 Click 'Analyze Freshness'
        3. 📊 View results and recommendations
        """)
        
        st.markdown("---")
        st.caption("🏆 Built for AIML+ Hackathon")
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded! Please ensure 'fruit_freshness_model.h5' is in the same directory.")
        st.info("📝 The model file will be added after training completes.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    # Determine model expected input size (height, width)
    try:
        m_input = model.input_shape  # e.g. (None, 240, 240, 3)
        if m_input and len(m_input) >= 3 and m_input[1] and m_input[2]:
            expected_size = (int(m_input[1]), int(m_input[2]))
        else:
            expected_size = (240, 240)  # fallback if model input shape is ambiguous
    except Exception:
        expected_size = (240, 240)

    st.caption(f"Model expects input size: {expected_size[0]} x {expected_size[1]} (HxW)")
    
    # Class names (based on your dataset)
    class_names = ['Fresh', 'Rotten']
    
    # File uploader
    st.markdown("### 📤 Upload Fruit Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the fruit"
    )
    
    if uploaded_file is not None:
        # Display columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption='Original Image', use_column_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"- Format: {image.format}")
            st.write(f"- Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- Mode: {image.mode}")
        
        with col2:
            st.markdown("#### 🔍 Analysis")
            
            # Analyze button
            if st.button('🚀 Analyze Freshness', type='primary', use_container_width=True):
                with st.spinner('🧠 AI is analyzing the image...'):
                    try:
                        # Preprocess image (resize to model's expected input size)
                        processed_img = preprocess_image(image, target_size=expected_size)

                        # Inform user if resizing occurred
                        try:
                            if image.size != (expected_size[1], expected_size[0]):
                                st.info(f"Uploaded image was resized from {image.size[0]}x{image.size[1]} to {expected_size[0]}x{expected_size[1]} for model input.")
                        except Exception:
                            pass
                        
                        # Make prediction
                        predictions = model.predict(processed_img, verbose=0)
                        
                        # Get predicted class
                        predicted_idx = np.argmax(predictions[0])
                        predicted_class = class_names[predicted_idx]
                        confidence = predictions[0][predicted_idx] * 100
                        
                        # Calculate freshness percentage
                        freshness_percentage, fresh_conf, rotten_conf = calculate_freshness_percentage(
                            predictions, predicted_class
                        )
                        
                        # Get freshness level
                        level, emoji, color = get_freshness_level(freshness_percentage)
                        
                        # Get recommendation
                        recommendation = get_recommendation(freshness_percentage)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Main prediction card
                        if predicted_class == "Fresh":
                            st.markdown(f"""
                                <div class="fresh-card">
                                    <h2>{emoji} {predicted_class}</h2>
                                    <h3>{confidence:.1f}% Confidence</h3>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="rotten-card">
                                    <h2>{emoji} {predicted_class}</h2>
                                    <h3>{confidence:.1f}% Confidence</h3>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Freshness metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Freshness Score",
                                value=f"{freshness_percentage:.1f}%",
                                delta=f"{level}"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Fresh Confidence",
                                value=f"{fresh_conf:.1f}%"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="Rotten Confidence",
                                value=f"{rotten_conf:.1f}%"
                            )
                        
                        # Gauge chart
                        st.markdown("#### 📊 Freshness Gauge")
                        fig = create_gauge_chart(freshness_percentage)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendation
                        st.markdown("#### 💡 Recommendation")
                        st.info(recommendation)
                        
                        # Detailed breakdown
                        with st.expander("📈 View Detailed Breakdown"):
                            st.markdown("**Prediction Probabilities:**")
                            for i, class_name in enumerate(class_names):
                                prob = predictions[0][i] * 100
                                # Streamlit progress requires a native Python int or float (not numpy types)
                                st.progress(float(prob) / 100)
                                st.caption(f"{class_name}: {prob:.2f}%")
                            
                            st.markdown("**Freshness Calculation Logic:**")
                            st.write(f"""
                            - Model predicted: **{predicted_class}**
                            - Prediction confidence: **{confidence:.1f}%**
                            - Freshness percentage: **{freshness_percentage:.1f}%**
                            - Fresh confidence: **{fresh_conf:.1f}%**
                            - Rotten confidence: **{rotten_conf:.1f}%**
                            """)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.exception(e)


if __name__ == "__main__":
    main()