import sys
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from data_loader import load_data, clean_initial_data
from preprocessor import TextPreprocessor
from model import get_model, EmotionDataset


EMOTION_COLUMNS = ['anger', 'confusion', 'fear', 'joy', 'sadness']

def train(data_path, output_dir, epochs=3, sample_size=None):

    print("Loading data...")
    df = load_data(data_path)
    if df.empty:
        print("No data found or data is empty.")
        return

    df = clean_initial_data(df)
    
    # Check if emotion columns exist, if not filter only those that exist
    available_emotions = [col for col in EMOTION_COLUMNS if col in df.columns]
    if not available_emotions:
        print(f"Error: None of the expected emotion columns {EMOTION_COLUMNS} found in dataset.")
        return
        
    if sample_size:
        print(f"Sampling {sample_size} rows for debugging...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess(df)
    
    X = df['text_processed']
    y = df[available_emotions]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Tokenizing...")
    train_encodings = preprocessor.tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    val_encodings = preprocessor.tokenizer(
        X_val.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    
    train_dataset = EmotionDataset(train_encodings, y_train)
    val_dataset = EmotionDataset(val_encodings, y_val)
    
    print("Initializing model...")
    model = get_model(len(available_emotions))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, 'logs'),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    preprocessor.tokenizer.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, help="Number of samples to train on")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    train(data_dir, models_dir, epochs=args.epochs, sample_size=args.sample)
