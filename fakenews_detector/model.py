from transformers import pipeline
import torch
import requests
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from .utils import clean_text, extract_claims, get_article_metadata

logging.basicConfig(level=logging.INFO)

class FakeNewsDetector:
    def __init__(self, model_name: str = "distilbert-base-uncased", fact_check_apis: Optional[Dict[str, str]] = None):
        try:
            # Initialize the text classification pipeline with a smaller model
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                device=-1 if not torch.cuda.is_available() else 0,
                max_length=512,
                truncation=True
            )
            
            # Initialize bias detection pipeline with a smaller model
            try:
                self.bias_detector = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    device=-1 if not torch.cuda.is_available() else 0,
                    max_length=512,
                    truncation=True
                )
            except Exception as e:
                logging.warning(f"Failed to load bias detection model: {str(e)}")
                self.bias_detector = None
            
            self.fact_check_apis = fact_check_apis or {}
            
            # Initialize fact-checking sources
            self.fact_checking_sources = {
                'google': 'https://factchecktools.googleapis.com/v1alpha1/claims:search',
                'politifact': 'https://www.politifact.com/api/v1/fact-checks/',
                'snopes': 'https://www.snopes.com/api/v1/fact-checks/',
                'reuters': 'https://www.reuters.com/fact-check/',
                'afp': 'https://factcheck.afp.com/api/v1/fact-checks/'
            }
            
            # Domain credibility database
            self.credible_domains = {
                'reuters.com': 0.95,
                'apnews.com': 0.95,
                'bbc.com': 0.9,
                'bbc.co.uk': 0.9,
                'npr.org': 0.9,
                'nytimes.com': 0.85,
                'washingtonpost.com': 0.85,
                'theguardian.com': 0.85,
                'wsj.com': 0.85,
                'economist.com': 0.85,
                'nature.com': 0.95,
                'science.org': 0.95,
                'scientificamerican.com': 0.9,
                'who.int': 0.95,
                'edu': 0.8,
                'gov': 0.85
            }
            
            # Load credibility indicators
            try:
                with open('data/credibility_indicators.json', 'r') as f:
                    self.credibility_indicators = json.load(f)
            except FileNotFoundError:
                self.credibility_indicators = {
                    "factual_language": [
                        "according to", "research shows", "studies indicate",
                        "evidence suggests", "data shows", "experts say"
                    ],
                    "emotional_language": [
                        "shocking", "outrageous", "unbelievable", "must see",
                        "won't believe", "amazing"
                    ],
                    "source_citation": [
                        "reported by", "published in", "cited in",
                        "as stated in", "according to sources"
                    ],
                    "conspiracy_language": [
                        "they don't want you to know", "wake up", "sheeple",
                        "conspiracy", "coverup", "hidden truth", "secret agenda"
                    ],
                    "clickbait_patterns": [
                        "you won't believe", "mind-blowing", "shocking truth",
                        "what they don't tell you", "secret trick"
                    ],
                    "scientific_language": [
                        "peer-reviewed", "clinical trial", "study published",
                        "research paper", "scientific evidence", "data analysis"
                    ]
                }
                
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        return clean_text(text)

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks that fit within model's max length."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Approximate token length (rough estimate)
            word_length = len(word.split())
            if current_length + word_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def analyze(self, text: str) -> Dict:
        """Perform comprehensive analysis of the text."""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Split text into chunks if necessary
        text_chunks = self.chunk_text(cleaned_text)
        
        # Process each chunk and aggregate results
        chunk_results = []
        chunk_scores = []
        for chunk in text_chunks:
            # Text classification for chunk
            classification_result = self.classifier(chunk)[0]
            chunk_results.append(classification_result)
            chunk_scores.append(classification_result['score'])
        
        # Aggregate results (use weighted average based on chunk scores)
        is_fake = sum(1 for result in chunk_results if result['label'] == 'LABEL_1') > len(chunk_results) / 2
        confidence = sum(score for score in chunk_scores) / len(chunk_scores)
        
        # Extract claims from text
        claims = extract_claims(cleaned_text)
        
        # Parallel processing of claims
        with ThreadPoolExecutor() as executor:
            fact_check_futures = [executor.submit(self.fact_check_claim, claim) for claim in claims]
            fact_checks = [future.result() for future in fact_check_futures]
        
        # Get bias analysis
        bias_score = 0.5  # Default neutral bias score
        if self.bias_detector is not None:
            try:
                # Process each chunk for bias
                bias_scores = []
                for chunk in text_chunks:
                    bias_result = self.bias_detector(chunk)[0]
                    bias_scores.append(bias_result['score'] if bias_result['label'] == 'LABEL_1' else 1 - bias_result['score'])
                bias_score = sum(bias_scores) / len(bias_scores)
            except Exception as e:
                logging.warning(f"Error in bias detection: {str(e)}")

        # Analyze credibility indicators
        credibility_scores = self.analyze_credibility_indicators(cleaned_text)
        
        # Get source reliability
        source_reliability = self.calculate_source_reliability(fact_checks)
        
        # Prepare key findings
        key_findings = self.generate_key_findings(fact_checks, credibility_scores, bias_score)
        
        # Determine verification status
        verification_status = self.determine_verification_status(fact_checks, credibility_scores)
        
        # Calculate bias level
        bias_level = self.determine_bias_level(bias_score)
        
        # Prepare sources
        sources = self.gather_sources(fact_checks)
        
        return {
            'label': 'FAKE' if is_fake else 'REAL',
            'confidence': confidence,
            'verification_status': verification_status,
            'bias_level': bias_level,
            'key_findings': key_findings,
            'fact_checks': fact_checks,
            'sources': sources,
            'credibility_score': np.mean(list(credibility_scores.values())),
            'bias_score': bias_score,
            'source_reliability': source_reliability,
            'claim_support': self.calculate_claim_support(fact_checks)
        }

    def analyze_source_credibility(self, url: str) -> Dict[str, float]:
        """Analyze the credibility of a source URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            base_domain = '.'.join(domain.split('.')[-2:])
            
            # Check domain credibility
            credibility_score = self.credible_domains.get(base_domain, 0.5)
            
            # Check for educational or government domains
            if domain.endswith('.edu'):
                credibility_score = max(credibility_score, self.credible_domains['edu'])
            elif domain.endswith('.gov'):
                credibility_score = max(credibility_score, self.credible_domains['gov'])
            
            return {
                'domain': domain,
                'credibility_score': credibility_score,
                'is_known_source': base_domain in self.credible_domains or any(domain.endswith(x) for x in ['.edu', '.gov'])
            }
        except Exception as e:
            logging.error(f"Error analyzing source credibility: {str(e)}")
            return {'domain': '', 'credibility_score': 0.5, 'is_known_source': False}

    def search_related_articles(self, claim: str) -> List[Dict]:
        """Search for related articles to verify the claim."""
        try:
            # Use a list of trusted news sources to search
            trusted_sources = [
                'https://www.reuters.com/fact-check/',
                'https://apnews.com/hub/ap-fact-check',
                'https://www.bbc.com/news/reality_check',
                'https://www.politifact.com/',
                'https://www.factcheck.org/'
            ]
            
            verified_results = []
            for source_url in trusted_sources:
                try:
                    # Get source credibility
                    credibility = self.analyze_source_credibility(source_url)
                    
                    verified_results.append({
                        'title': f"Fact Check from {credibility['domain']}",
                        'url': source_url,
                        'snippet': f"Trusted fact-checking source with {credibility['credibility_score']*100:.0f}% credibility rating",
                        'source_credibility': credibility
                    })
                except Exception as e:
                    logging.warning(f"Error processing source {source_url}: {str(e)}")
                    continue
            
            return verified_results[:5]  # Return top 5 sources
        except Exception as e:
            logging.error(f"Error searching related articles: {str(e)}")
            return []

    def fact_check_claim(self, claim: str) -> Dict:
        """Enhanced fact checking with multiple sources and credibility analysis."""
        results = []
        
        # Search related articles
        related_articles = self.search_related_articles(claim)
        
        # Check against fact-checking APIs
        for source, api_key in self.fact_check_apis.items():
            if not api_key:
                continue
                
            try:
                url = self.fact_checking_sources[source]
                headers = {'Authorization': f'Bearer {api_key}'}
                params = {'query': claim}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                if source == 'google':
                    results.extend(self._parse_google_factcheck(data))
                elif source == 'politifact':
                    results.extend(self._parse_politifact(data))
                elif source == 'snopes':
                    results.extend(self._parse_snopes(data))
                    
            except Exception as e:
                logging.error(f"Error checking {source}: {str(e)}")
                continue
        
        # Aggregate results with related articles
        aggregated_result = self._aggregate_fact_checks(results, claim)
        aggregated_result['related_articles'] = related_articles
        
        return aggregated_result

    def analyze_credibility_indicators(self, text: str) -> Dict[str, float]:
        """Analyze text for credibility indicators."""
        scores = {}
        
        for indicator, patterns in self.credibility_indicators.items():
            score = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    score += 1
            
            scores[indicator] = score / total_patterns if total_patterns > 0 else 0
            
        return scores

    def determine_verification_status(self, fact_checks: List[Dict], credibility_scores: Dict[str, float]) -> str:
        """Determine overall verification status based on fact checks and credibility."""
        if not fact_checks:
            return "UNVERIFIED"
            
        true_claims = sum(1 for check in fact_checks if check['status'].lower() == 'true')
        false_claims = sum(1 for check in fact_checks if check['status'].lower() == 'false')
        
        avg_credibility = np.mean(list(credibility_scores.values()))
        
        if false_claims > true_claims and avg_credibility < 0.5:
            return "FALSE"
        elif true_claims > false_claims and avg_credibility > 0.7:
            return "VERIFIED"
        else:
            return "MIXED"

    def determine_bias_level(self, bias_score: float) -> str:
        """Determine bias level based on bias score."""
        if bias_score < 0.3:
            return "LOW"
        elif bias_score < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    def calculate_source_reliability(self, fact_checks: List[Dict]) -> float:
        """Calculate source reliability score."""
        if not fact_checks:
            return 0.0
            
        reliable_sources = sum(1 for check in fact_checks if check.get('source_reliability', 0) > 0.7)
        return reliable_sources / len(fact_checks)

    def calculate_claim_support(self, fact_checks: List[Dict]) -> float:
        """Calculate claim support score based on fact checks."""
        if not fact_checks:
            return 0.0
            
        supported_claims = sum(1 for check in fact_checks if check['status'].lower() == 'true')
        return supported_claims / len(fact_checks)

    def generate_key_findings(self, fact_checks: List[Dict], credibility_scores: Dict[str, float], bias_score: float) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        # Add findings based on fact checks
        if fact_checks:
            true_ratio = sum(1 for check in fact_checks if check['status'].lower() == 'true') / len(fact_checks)
            if true_ratio < 0.3:
                findings.append("Multiple claims in this content have been fact-checked as false")
            elif true_ratio > 0.7:
                findings.append("Most claims in this content are supported by fact-checks")
        
        # Add findings based on credibility indicators
        avg_credibility = np.mean(list(credibility_scores.values()))
        if avg_credibility < 0.3:
            findings.append("Content shows multiple signs of low credibility")
        elif avg_credibility > 0.7:
            findings.append("Content demonstrates strong credibility indicators")
        
        # Add findings based on bias
        if bias_score > 0.7:
            findings.append("High level of potential bias detected in the content")
        elif bias_score < 0.3:
            findings.append("Content appears to be relatively unbiased")
            
        return findings

    def gather_sources(self, fact_checks: List[Dict]) -> List[Dict]:
        """Gather and format sources from fact checks."""
        sources = []
        seen_urls = set()
        
        for check in fact_checks:
            if check.get('source') and check['source'].get('url') not in seen_urls:
                sources.append({
                    'title': check['source'].get('name', ''),
                    'url': check['source'].get('url', ''),
                    'description': check['source'].get('description', '')
                })
                seen_urls.add(check['source'].get('url'))
                
        return sources

    def _parse_google_factcheck(self, data: Dict) -> List[Dict]:
        """Parse Google Fact Check API response."""
        results = []
        for claim in data.get('claims', []):
            results.append({
                'claim': claim.get('text', ''),
                'status': claim.get('claimReview', [{}])[0].get('textualRating', ''),
                'explanation': claim.get('claimReview', [{}])[0].get('textualSummary', ''),
                'source': {
                    'name': claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', ''),
                    'url': claim.get('claimReview', [{}])[0].get('url', ''),
                    'description': ''
                },
                'source_reliability': 0.9  # Google-verified fact checkers
            })
        return results

    def _parse_politifact(self, data: Dict) -> List[Dict]:
        """Parse PolitiFact API response."""
        results = []
        for fact in data.get('results', []):
            results.append({
                'claim': fact.get('statement', ''),
                'status': fact.get('ruling', ''),
                'explanation': fact.get('ruling_explanation', ''),
                'source': {
                    'name': 'PolitiFact',
                    'url': fact.get('url', ''),
                    'description': 'Pulitzer Prize-winning fact-checking website'
                },
                'source_reliability': 0.95
            })
        return results

    def _parse_snopes(self, data: Dict) -> List[Dict]:
        """Parse Snopes API response."""
        results = []
        for fact in data.get('fact_checks', []):
            results.append({
                'claim': fact.get('claim', ''),
                'status': fact.get('rating', ''),
                'explanation': fact.get('explanation', ''),
                'source': {
                    'name': 'Snopes',
                    'url': fact.get('url', ''),
                    'description': 'One of the first online fact-checking websites'
                },
                'source_reliability': 0.9
            })
        return results

    def _aggregate_fact_checks(self, results: List[Dict], original_claim: str) -> Dict:
        """Aggregate and summarize fact check results."""
        if not results:
            return {
                'claim': original_claim,
                'status': 'UNVERIFIED',
                'explanation': 'No fact checks found for this claim.',
                'source': None
            }
            
        # Sort by source reliability
        results.sort(key=lambda x: x.get('source_reliability', 0), reverse=True)
        
        # Return the most reliable fact check
        return results[0] 