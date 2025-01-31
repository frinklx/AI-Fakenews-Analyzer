from flask import Flask, request, jsonify, render_template
from .model import FakeNewsDetector
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple
import logging
import atexit
import torch
import gc
import threading
import traceback

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Thread-safe model manager with lazy loading
class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance.model = None
                    cls._instance.initialized = False
                    cls._instance.error = None
        return cls._instance
    
    def initialize(self):
        if self.initialized:
            return self.model
            
        with self._lock:
            if not self.initialized:
                try:
                    # Clear any existing memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Initialize model with error handling
                    self.model = FakeNewsDetector(
                        model_name="roberta-large-mnli",
                        fact_check_apis={
                            "google": os.getenv("GOOGLE_FACT_CHECK_API_KEY"),
                            "politifact": os.getenv("POLITIFACT_API_KEY"),
                            "snopes": os.getenv("SNOPES_API_KEY")
                        }
                    )
                    self.initialized = True
                    self.error = None
                except Exception as e:
                    self.error = str(e)
                    logging.error(f"Error initializing model: {str(e)}\n{traceback.format_exc()}")
                    raise
            return self.model
    
    def get_model(self):
        try:
            if not self.initialized:
                return self.initialize()
            return self.model
        except Exception as e:
            logging.error(f"Error getting model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def cleanup(self):
        with self._lock:
            if self.model is not None:
                try:
                    # Clear model resources
                    if hasattr(self.model, 'classifier'):
                        del self.model.classifier
                    if hasattr(self.model, 'bias_detector'):
                        del self.model.bias_detector
                    self.model = None
                    self.initialized = False
                    self.error = None
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error during cleanup: {str(e)}\n{traceback.format_exc()}")
                    raise

model_manager = ModelManager()

def cleanup():
    """Cleanup resources at shutdown"""
    try:
        model_manager.cleanup()
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}\n{traceback.format_exc()}")

# Register cleanup function
atexit.register(cleanup)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    """Check model status"""
    try:
        model = model_manager.get_model()
        return jsonify({
            'status': 'ready' if model else 'not_initialized',
            'error': model_manager.error
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def fetch_article_content(url: str) -> Tuple[str, Dict]:
    """Fetch article content and metadata from URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article text
        paragraphs = []
        for p in soup.find_all(['p', 'article', 'div.article-content']):
            text = p.get_text().strip()
            if len(text) > 50:  # Only include substantial paragraphs
                paragraphs.append(text)
        
        # Limit total text length to avoid memory issues
        article_text = ' '.join(paragraphs)
        if len(article_text) > 10000:  # Limit to ~10k characters
            article_text = article_text[:10000] + "..."
        
        # Extract metadata
        metadata = {
            'title': soup.title.string if soup.title else '',
            'published_date': soup.find('meta', {'property': 'article:published_time'})['content'] if soup.find('meta', {'property': 'article:published_time'}) else '',
            'author': soup.find('meta', {'name': 'author'})['content'] if soup.find('meta', {'name': 'author'}) else '',
            'source': url,
            'truncated': len(article_text) > 10000
        }
        
        return article_text, metadata
    except Exception as e:
        logging.error(f"Error fetching article: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Failed to fetch article from URL: {str(e)}")

def get_similar_claims(claim: str) -> List[Dict]:
    """Fetch similar fact-checked claims from various sources."""
    similar_claims = []
    model = model_manager.get_model()
    
    try:
        # Query fact-checking APIs
        for source in ['google', 'politifact', 'snopes']:
            if model.fact_check_apis.get(source):
                claims = model.query_fact_check_api(source, claim)
                similar_claims.extend(claims)
        
        # Deduplicate and sort by relevance
        similar_claims = list({claim['claim']: claim for claim in similar_claims}.values())
        similar_claims.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        return similar_claims[:5]  # Return top 5 similar claims
    except Exception as e:
        logging.error(f"Error fetching similar claims: {str(e)}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'Empty request'}), 400
            
        text = ''
        metadata = {}
        
        if 'url' in data:
            text, metadata = fetch_article_content(data['url'])
        elif 'text' in data:
            text = data['text']
        else:
            return jsonify({'error': 'No URL or text provided'}), 400

        if not text.strip():
            return jsonify({'error': 'Empty text content'}), 400

        # Get model instance
        try:
            model = model_manager.get_model()
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
        except Exception as e:
            logging.error(f"Error getting model: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'Failed to load model'}), 500

        # Perform analysis
        try:
            analysis_result = model.analyze(text)
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'Analysis failed'}), 500
        
        # Get similar fact-checked claims
        similar_claims = get_similar_claims(text)
        
        # Prepare response
        response = {
            'classification': analysis_result['label'],
            'confidence': round(analysis_result['confidence'] * 100, 2),
            'verification_status': analysis_result['verification_status'],
            'bias_level': analysis_result['bias_level'],
            'key_findings': analysis_result['key_findings'],
            'fact_checks': analysis_result['fact_checks'],
            'sources': [
                {
                    'title': source['title'],
                    'url': source['url'],
                    'description': source['description']
                }
                for source in analysis_result['sources']
            ],
            'detailed_scores': {
                'Credibility': round(analysis_result['credibility_score'] * 100, 2),
                'Bias': round(analysis_result['bias_score'] * 100, 2),
                'Source Reliability': round(analysis_result['source_reliability'] * 100, 2),
                'Claim Support': round(analysis_result['claim_support'] * 100, 2)
            },
            'similar_claims': similar_claims,
            'metadata': metadata
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=False) 