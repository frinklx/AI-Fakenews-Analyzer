from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fakenews_detector",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered fake news detection system using zero-shot classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fakenews",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.3.3",
        "werkzeug>=2.3.7",
        "transformers>=4.30.2",
        "torch>=2.0.1",
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "scikit-learn>=1.2.2",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "beautifulsoup4>=4.12.2",
        "newspaper3k>=0.2.8",
        "spacy>=3.5.3",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.3",
        "accelerate>=0.20.3"
    ],
    entry_points={
        "console_scripts": [
            "fake-news-detector=app:main",
        ],
    },
) 