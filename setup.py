"""Setup script for MAVYN package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mavyn",
    version="2.1.0",
    author="Mahir",
    author_email="aislam192054@gmail.com",
    description="Local-first paper manager with semantic search and LLM reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alMohimanul/mavyn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "click>=8.1.0",
        "pypdf2>=3.0.0",
        "pdfplumber>=0.11.0",
        "sentence-transformers>=2.3.0",
        "faiss-cpu>=1.7.4",
        "sqlalchemy>=2.0.0",
        "groq>=0.4.0",
        "google-generativeai>=0.3.0",
        "httpx>=0.26.0",
        "rich>=13.7.0",
        "python-dotenv>=1.0.0",
        "python-docx>=0.8.11",
    ],
    entry_points={
        "console_scripts": [
            "mavyn=MAVYN.cli.commands:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
            "pre-commit>=3.5.0",
            "build>=1.0.0",
            "twine>=4.0.0",
        ],
    },
)
