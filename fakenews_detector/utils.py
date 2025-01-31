from newspaper import Article
import logging
import re
import spacy
from typing import List, Dict
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

def clean_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def extract_claims(text: str) -> List[str]:
    """Extract potential claims from text using NLP."""
    doc = nlp(text)
    claims = []
    
    # Extract sentences that are likely to be claims
    for sent in doc.sents:
        # Skip very short sentences
        if len(sent.text.split()) < 4:
            continue
            
        # Look for claim indicators
        text = sent.text.lower()
        if any(indicator in text for indicator in [
            "according to",
            "claimed",
            "stated",
            "reported",
            "said",
            "suggests",
            "shows",
            "proves",
            "demonstrates",
            "confirms"
        ]):
            claims.append(sent.text)
            continue
            
        # Look for factual statements
        if any(token.pos_ in ["VERB"] for token in sent):
            if any(token.dep_ in ["nsubj", "nsubjpass"] for token in sent):
                claims.append(sent.text)
                
    return claims

def get_article_metadata(url: str) -> Dict:
    """Extract metadata from article URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        metadata = {
            'title': None,
            'author': None,
            'date_published': None,
            'source_domain': None,
            'description': None
        }
        
        # Extract title
        metadata['title'] = (
            soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') 
            else soup.title.string if soup.title 
            else None
        )
        
        # Extract author
        author_meta = (
            soup.find('meta', {'name': 'author'}) or 
            soup.find('meta', {'property': 'article:author'})
        )
        metadata['author'] = author_meta['content'] if author_meta else None
        
        # Extract publication date
        date_meta = (
            soup.find('meta', {'property': 'article:published_time'}) or
            soup.find('meta', {'name': 'publication_date'})
        )
        if date_meta and date_meta.get('content'):
            try:
                date = datetime.fromisoformat(date_meta['content'].replace('Z', '+00:00'))
                metadata['date_published'] = date.isoformat()
            except ValueError:
                pass
        
        # Extract domain
        metadata['source_domain'] = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
        
        # Extract description
        description_meta = (
            soup.find('meta', {'name': 'description'}) or
            soup.find('meta', {'property': 'og:description'})
        )
        metadata['description'] = description_meta['content'] if description_meta else None
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {str(e)}")
        return metadata

def analyze_writing_style(text: str) -> Dict[str, float]:
    """Analyze writing style indicators."""
    doc = nlp(text)
    
    # Calculate various style metrics
    total_sentences = len(list(doc.sents))
    if total_sentences == 0:
        return {
            'avg_sentence_length': 0,
            'complexity_score': 0,
            'emotional_language': 0
        }
    
    # Average sentence length
    avg_sentence_length = len([token for token in doc if not token.is_punct]) / total_sentences
    
    # Complexity score (based on word length and rare words)
    complex_words = len([token for token in doc if len(token.text) > 6 and token.is_alpha])
    complexity_score = complex_words / len([token for token in doc if token.is_alpha]) if len([token for token in doc if token.is_alpha]) > 0 else 0
    
    # Emotional language score
    emotional_words = len([token for token in doc if token.pos_ == "ADJ" or token.pos_ == "ADV"])
    emotional_score = emotional_words / len([token for token in doc if token.is_alpha]) if len([token for token in doc if token.is_alpha]) > 0 else 0
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'complexity_score': complexity_score,
        'emotional_language': emotional_score
    }

def check_source_credibility(domain: str) -> Dict[str, float]:
    """Check credibility of news source."""
    # This would typically involve checking against a database of known sources
    # For now, we'll return placeholder scores
    return {
        'reliability_score': 0.5,  # Default neutral score
        'bias_score': 0.5,
        'factual_reporting_score': 0.5
    } 