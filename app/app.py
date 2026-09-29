import streamlit as st
import sys
from pathlib import Path

# -----------------------------------
# Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src folder to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

from predict import predict_image


# -----------------------------------
# Streamlit page configuration
# -----------------------------------

st.set_page_config(
    page_title="Illegal Mining Detection",
    page_icon="🛰️",
    layout="wide"
)

# -----------------------------------
# Custom Styling
# -----------------------------------

st.markdown("""
<style>

/* ================================
   GLOBAL LAYOUT
================================ */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef6f3 50%,
        #f8fafc 100%
    );
}

.block-container {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem clamp(1rem, 4vw, 4rem) 4rem;
    box-sizing: border-box;
}


/* ================================
   HERO
================================ */

.hero {
    width: 100%;
    box-sizing: border-box;

    padding: clamp(2rem, 4vw, 3.5rem);

    margin-top: 1.5rem;
    margin-bottom: 2.5rem;

    border-radius: 28px;

    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #164e63 100%
    );

    border: 1px solid rgba(255, 255, 255, 0.08);

    box-shadow:
        0 20px 45px rgba(15, 23, 42, 0.18);

    overflow: hidden;

    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: clamp(28px, 4vw, 52px);
    font-weight: 800;

    line-height: 1.12;

    color: white;

    margin: 0 0 1rem 0;

    max-width: 100%;

    overflow-wrap: anywhere;
}

.hero-subtitle {
    font-size: clamp(15px, 1.6vw, 20px);

    line-height: 1.6;

    color: #cbd5e1;

    max-width: 800px;
}


/* ================================
   INFORMATION CARDS
================================ */

.info-card {
    width: 100%;
    box-sizing: border-box;

    min-height: 170px;

    padding: clamp(1.2rem, 2vw, 1.8rem);

    border-radius: 20px;

    background: rgba(255, 255, 255, 0.95);

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.07);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.info-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 14px 30px rgba(15, 23, 42, 0.12);
}

.card-title {
    font-size: clamp(16px, 1.5vw, 20px);

    font-weight: 700;

    color: #0f172a;

    margin-bottom: 0.7rem;
}

.card-text {
    font-size: clamp(14px, 1.3vw, 17px);

    color: #475569;

    line-height: 1.6;
}


/* ================================
   UPLOAD AREA
================================ */

[data-testid="stFileUploader"] {
    width: 100%;
    box-sizing: border-box;

    background: rgba(255, 255, 255, 0.95);

    padding: clamp(1rem, 2vw, 1.8rem);

    border-radius: 20px;

    border: 2px dashed #94a3b8;

    transition:
        border-color 0.2s ease,
        background 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #0f766e;

    background: #f8fffd;
}


/* ================================
   BUTTONS
================================ */

.stButton > button {

    width: 100%;

    min-height: 3rem;

    border-radius: 12px;

    font-weight: 700;

    border: none;

    background: #0f766e;

    color: white;

    transition:
        transform 0.2s ease,
        background 0.2s ease;
}

.stButton > button:hover {

    background: #115e59;

    transform: translateY(-2px);
}


/* ================================
   METRICS
================================ */

[data-testid="stMetric"] {

    width: 100%;
    box-sizing: border-box;

    background: rgba(255, 255, 255, 0.95);

    padding: 1.3rem;

    border-radius: 18px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 6px 20px rgba(15, 23, 42, 0.06);
}


/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] {

    background: #f1f5f9;

}

section[data-testid="stSidebar"] > div {

    padding-top: 2rem;

}


/* ================================
   IMAGES
================================ */

img {

    max-width: 100% !important;

    height: auto !important;

    border-radius: 16px;
}


/* ================================
   MOBILE / SMALL SCREENS
================================ */

@media (max-width: 900px) {

    .block-container {

        padding:
            1.2rem
            1rem
            3rem;
    }

    .hero {

        padding: 1.8rem;

        border-radius: 18px;
    }

    .info-card {

        min-height: auto;

        margin-bottom: 1rem;
    }
}


/* ================================
   VERY SMALL SCREENS
================================ */

@media (max-width: 600px) {

    .block-container {

        padding:
            1rem
            0.8rem
            2rem;
    }

    .hero {

        padding: 1.4rem;

        border-radius: 16px;
    }

    .hero-title {

        font-size: 28px;

        line-height: 1.15;
    }

    .hero-subtitle {

        font-size: 15px;
    }

    .info-card {

        padding: 1.2rem;

        border-radius: 16px;
    }

}


/* ================================
   REMOVE HORIZONTAL OVERFLOW
================================ */

html,
body {

    max-width: 100%;

    overflow-x: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.header("About the System")

    st.write(
        "This application uses a ResNet18 deep learning model "
        "to classify aerial and satellite images into Mining "
        "and Non-Mining visual classes."
    )

    st.divider()

    st.subheader("Model")

    st.write("**Architecture:** ResNet18")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 2")
    st.write("**Device:** CPU")

    st.divider()

    st.subheader("Classes")

    st.write("**Mining**")
    st.write("**Non-Mining**")
    
    st.divider()

    st.subheader("Dataset")

    st.write("Total Images: 2,006")
    st.write("Mining: 1,010")
    st.write("Non-Mining: 996")

    st.divider()

    st.subheader("Test Performance")

    st.write("Accuracy: 100.00%")
    st.write("Precision: 100.00%")
    st.write("Recall: 100.00%")
    st.write("F1-Score: 100.00%")

    st.divider()

    st.caption(
        "This system provides image-based classification "
        "and does not independently determine the legal "
        "status of mining activity."
    )


# -----------------------------------
# Title
# -----------------------------------

st.html("""
<div class="hero">

    <div class="hero-title">
        ILLEGAL MINING DETECTION USING AERIAL IMAGES
    </div>

    <div class="hero-subtitle">
        Model for Aerial & Satellite Image Analysis
        <br>
        Powered by ResNet18
    </div>

</div>
""")

st.write(
    "Upload an aerial or satellite image to classify "
    "model to analyze visual patterns associated with mining activity using ResNet18."
)

col1, col2, col3 = st.columns(
    [1, 1, 1],
    gap="large"
)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">🧠 Deep Learning</div>
        <div class="card-text">
            ResNet18 trained for aerial image classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">🛰️ Image Analysis</div>
        <div class="card-text">
            Analyze aerial and satellite imagery for mining-related
            visual patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="card-title">📊 AI Prediction</div>
        <div class="card-text">
            Receive a classification and model confidence score.
        </div>
    </div>
    """, unsafe_allow_html=True)


st.info(
    "This system identifies visual patterns associated with "
    "the dataset's Mining and Non-Mining classes. "
    "It does not independently determine the legal status of mining activity."
)


# -----------------------------------
# File uploader
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)


# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    from PIL import Image

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width="stretch"
    )

    # Prediction button
    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing image..."):

            prediction, confidence = predict_image(image)
            
        st.subheader("🔍 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                prediction
            )

        with col2:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        st.divider()

        if prediction == "Mining":

            st.warning(
                "⚠️ Potential Mining Activity Detected"
            )

            st.write(
                "The model detected visual patterns associated "
                "with the Mining class in the uploaded image."
            )

        else:

            st.success(
                "✅ Non-Mining Pattern Detected"
            )

            st.write(
                "The model did not detect visual patterns "
                "associated with the Mining class."
            )

        st.write("### Confidence Level")

        st.progress(
            min(int(confidence), 100)
        )

        st.caption(
            "The confidence value represents the model's "
            "classification confidence. It should not be "
            "interpreted as proof of illegal activity."
        )

    