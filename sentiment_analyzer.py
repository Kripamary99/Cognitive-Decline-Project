from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import streamlit as st

class SentimentAnalyzer:
    def __init__(self):
        # Load model and tokenizer
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            return tokenizer, model
        
        self.tokenizer, self.model = load_model()
        self.model.eval()
    
    def analyze_text(self, text):
        """
        Analyze sentiment of text using BERT.
        Returns overall sentiment and sentence-level analysis.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Analyze each sentence
        sentence_results = []
        overall_sentiment = 0
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Tokenize and get sentiment
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get predicted sentiment (1-5 scale)
            sentiment_score = torch.argmax(predictions).item() + 1
            confidence_score = torch.max(predictions).item()
            
            # Convert to -1 to 1 scale
            normalized_score = (sentiment_score - 3) / 2
            
            sentence_results.append({
                'sentence': sentence,
                'sentiment': sentiment_score,
                'confidence': confidence_score,
                'normalized_score': normalized_score
            })
            
            overall_sentiment += normalized_score
        
        # Calculate overall sentiment
        if sentence_results:
            overall_sentiment /= len(sentence_results)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentence_analysis': sentence_results
        }
    
    def get_sentiment_color(self, score):
        """Return color based on sentiment score."""
        if score > 0.2:
            return "#2ecc71"  # Green for positive
        elif score < -0.2:
            return "#e74c3c"  # Red for negative
        return "#f1c40f"  # Yellow for neutral
    
    def get_sentiment_emoji(self, score):
        """Return emoji based on sentiment score."""
        if score > 0.2:
            return "😊"
        elif score < -0.2:
            return "😞"
        return "😐"
    
    def get_sentiment_label(self, score):
        """Return text label based on sentiment score."""
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        return "Neutral" 