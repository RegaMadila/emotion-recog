import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.predictor import Predictor


st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box_shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Emotion Recognition")
st.markdown("Enter text below to analyze the underlying emotions.")


st.sidebar.header("About")
st.sidebar.info(
    "This application uses a fine-tuned BERT model to detect emotions in text. "
    "Supported emotions: Anger, Confusion, Fear, Joy, Sadness."
)


@st.cache_resource
def load_predictor():
    model_path = os.path.join("models")
    if not os.path.exists(model_path):
        return None
    try:
        return Predictor(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_predictor()

if predictor is None:
    st.warning("Model not found in `models/` directory. Please train the model first.")
else:

    text_input = st.text_area("How are you feeling?", height=150, placeholder="Type something here...")

    if st.button("Analyze Emotion"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    results = predictor.predict(text_input)
                    
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:

                        top_emotion = max(results, key=results.get)
                        confidence = results[top_emotion]
                        st.metric(label="Dominant Emotion", value=top_emotion.capitalize(), delta=f"{confidence:.2%}")
                    
                    with col2:

                        df_results = pd.DataFrame(list(results.items()), columns=['Emotion', 'Score'])
                        df_results['Emotion'] = df_results['Emotion'].apply(lambda x: x.capitalize())
                        
                        fig, ax = plt.subplots()
                        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
                        ax.barh(df_results['Emotion'], df_results['Score'], color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Confidence Score")
                        st.pyplot(fig)
                        
                    with st.expander("View Detailed Scores"):
                        st.table(df_results.set_index('Emotion').style.format("{:.2%}"))
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text to analyze.")
