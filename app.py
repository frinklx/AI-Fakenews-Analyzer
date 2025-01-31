from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from newspaper import Article
import numpy as np
import re
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the zero-shot classification pipeline
try:
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    classifier = None

def clean_text(text):
    """Clean and preprocess the input text."""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def extract_article_content(url):
    """Extract article content from URL using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'title': article.title,
            'text': article.text,
            'success': True
        }
    except Exception as e:
        logging.error(f"Error extracting article: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def analyze_text(text):
    """Analyze text using the zero-shot classification model."""
    if not classifier:
        return {"error": "Model not loaded"}
    
    # Define the categories for classification
    candidate_labels = [
        "factual news",
        "fake news",
        "opinion piece",
        "satire"
    ]
    
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Perform classification
        result = classifier(cleaned_text, candidate_labels)
        
        # Calculate confidence scores
        scores = dict(zip(result['labels'], result['scores']))
        
        # Additional analysis
        analysis = {
            'classification': result['labels'][0],
            'confidence': round(result['scores'][0] * 100, 2),
            'detailed_scores': {
                label: round(score * 100, 2)
                for label, score in scores.items()
            },
            'length': len(text.split()),
            'success': True
        }
        
        return analysis
    
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if 'url' in data:
        # Handle URL input
        article_data = extract_article_content(data['url'])
        if not article_data['success']:
            return jsonify({'error': 'Failed to extract article content'})
        text = article_data['text']
    elif 'text' in data:
        # Handle direct text input
        text = data['text']
    else:
        return jsonify({'error': 'No text or URL provided'})
    
    # Perform analysis
    result = analyze_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 