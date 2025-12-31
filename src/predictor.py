import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from .preprocessor import TextPreprocessor

class Predictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist. Please train the model first.")
            
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Hardcoded for now, ideally saved with config
        self.emotion_labels = ['anger', 'confusion', 'fear', 'joy', 'sadness']

    def predict(self, text):
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.replace_emoticons(text)
        
        inputs = self.tokenizer(
            processed_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.sigmoid(outputs.logits)
            
        scores = predictions[0].cpu().numpy()
        results = {label: float(score) for label, score in zip(self.emotion_labels, scores)}
        
        return results
