CS410 - Fall 2024 - Final Project

Enhance OpenSearch capability with AI

Overview
This project implements an interactive chatbot that searches and analyzes AWS service logs using BM25 search algorithm and natural language processing. The chatbot provides an intuitive interface for querying log data and returns relevant log entries based on user queries.

Features
Interactive chat interface using Streamlit

BM25-based log search functionality

Caching mechanism for faster repeated queries

Natural language query processing

Real-time log analysis and parsing

Support for various AWS service log formats


dependencies:
  - python=3.10
  - conda-forge::openjdk=21
  - conda-forge::maven
  - conda-forge::lightgbm
  - nmslib
  - pytorch
  - faiss-cpu
  - pytorch
  - conda-forge::jupyterlab
  - conda-forge::notebook
  - conda-forge::ipywidgets
  - pip
  - pip:
    - streamlit
    - langchain
    - python-dotenv
    - pyserini==0.38.0
    - matplotlib
    - google-generativeai
    - langchain-google-genai
    - numpy<2


Installation

Create conda environment using environment.yml or requirement.txt file.
Activate conda environment.
Generate an API key using https://ai.google.dev/gemini-api/docs/api-key and copy the api key.
Export API key for the environment - export GOOGLE_API_KEY = ‘your api key here’.

Set up environment variables: Create a .env file in the project root and add:




Project Structure
project/
├── code/
│   ├── chatbot3.py
│   └── [other source files]
├── data/
│   └── logs/
│       └── aws-service-logs/
├── processed_corpus/
├── indexes/
├── requirements.txt
└── README.md


