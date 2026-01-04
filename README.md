# Emotion Recognition

This project demonstrates a complete end-to-end pipeline for Emotion Recognition using BERT and Streamlit.

## Model
The project uses `bert-base-uncased` from Hugging Face Transformers. It is fine-tuned to classify text into 5 dominant emotions:
- Anger
- Confusion
- Fear
- Joy
- Sadness

## Directory Structure
```
emotion_recog/
├── data/                   # Place your CSV datasets here
├── models/                 # Trained models are saved here
├── src/
│   ├── data_loader.py      # Data ingestion
│   ├── preprocessor.py     # Text cleaning and tokenization
│   ├── model.py            # BERT model definition
│   ├── train.py            # Training loop
│   └── predictor.py        # Inference class
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Place your emotion dataset CSV files in the `data/` directory.

3.  **Train the Model**:
    ```bash
    python src/train.py
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

tes webhook 2 discord
