# AI Fake News Detector

A modern web application that uses advanced AI to detect and analyze potential misinformation in news articles and text content.

![AI Fake News Detector](screenshot.png)

## Features

- 🔍 **URL Analysis**: Analyze news articles directly from URLs
- 📝 **Text Analysis**: Analyze any text content for potential misinformation
- 🎯 **Credibility Scoring**: Advanced AI-powered credibility assessment
- ⚖️ **Bias Detection**: Sophisticated bias analysis
- 🔒 **Source Reliability**: Evaluation of source trustworthiness
- ✅ **Fact Checking**: Cross-reference with known fact-checking sources
- 🔄 **Similar Claims**: Find and analyze related claims
- 🌙 **Dark Mode**: Beautiful dark theme with modern UI

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Python, Flask
- **AI/ML**: Hugging Face Transformers, DistilBERT
- **Styling**: TailwindCSS
- **Icons**: Heroicons

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/fakenews.git
cd fakenews
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python run.py
```

The application will be available at `http://localhost:5001`

## Usage

1. Choose analysis type (URL or Text)
2. Enter the content to analyze
3. Click "Analyze Content"
4. View detailed results including:
   - Credibility Score
   - Bias Analysis
   - Source Reliability
   - Fact Check Results
   - Similar Claims

## API Reference

### Analyze Endpoint

```http
POST /analyze
```

#### Request Body

```json
{
  "url": "https://example.com/article"
}
```
or
```json
{
  "text": "Article content to analyze"
}
```

#### Response

```json
{
  "credibility_score": 0.85,
  "bias_level": "LOW",
  "classification": "REAL",
  "confidence": 85.5,
  "fact_checks": [...],
  "similar_claims": [...],
  "detailed_scores": {
    "source_reliability": 0.75
  }
}
```

## Project Structure

```
fakenews/
├── fakenews_detector/
│   ├── __init__.py
│   ├── app.py
│   ├── model.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── styles.css
├── requirements.txt
├── run.py
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Performance Optimizations

- Optimized animations for smooth performance
- Reduced unnecessary DOM updates
- Efficient error handling
- Optimized image and asset loading
- Reduced console logging in production

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [TailwindCSS](https://tailwindcss.com/) for styling utilities
- [Heroicons](https://heroicons.com/) for beautiful icons

## Contact

frinklx - [@frinklx](https://x.com/frinklx)

Project Link: [https://github.com/frinklx/AI-Fakenews-Analyzer](https://github.com/frinklx/AI-Fakenews-Analyzer) 