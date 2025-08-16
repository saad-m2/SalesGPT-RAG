# SalesGPT-RAG

A Retrieval-Augmented Generation (RAG) platform that ingests a client's technical B2B product documentation, CRM data, and market insights to generate personalized sales pitches, use-case scenarios, and competitive talking points — complete with provenance and analytics.

## Project Overview

SalesGPT-RAG is an AI-powered B2B sales intelligence platform designed to help sales development representatives (SDRs) create personalized pitches and insights. The system combines:

- **Document Processing**: PDF parsing and text extraction
- **Vector Search**: Semantic similarity using HuggingFace embeddings
- **Hybrid Retrieval**: BM25 + dense vector search for optimal results
- **AI Generation**: LLM-powered pitch generation using Groq API
- **Lead Analytics**: Machine learning insights from CRM data
- **Streamlit Interface**: Easy-to-use web application

## Features

### 🚀 Core Functionality
- **Document Ingestion**: Upload PDFs and automatically chunk, embed, and index them
- **Smart Search**: Hybrid retrieval combining semantic similarity and keyword matching
- **AI Pitch Generation**: Generate personalized sales pitches, demo scripts, and use-case briefs
- **Lead Intelligence**: Analyze CRM data to identify patterns and segment leads
- **Source Attribution**: Always see where generated content comes from

### 📊 Analytics & Insights
- **Lead Clustering**: K-means clustering to identify customer segments
- **Engagement Prediction**: Random Forest classifier for lead scoring
- **Performance Metrics**: Recall@K evaluation for search quality

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Processing     │    │   Vector DB     │
│   (PDFs, CRM)  │───▶│  (Chunking,     │───▶│   (Qdrant)      │
└─────────────────┘    │   Embedding)    │    └─────────────────┘
                       └─────────────────┘              │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│   RAG Pipeline  │◀───│   Hybrid Search │
│   (User Input)  │    │  (LLM + RAG)    │    │  (BM25 + Dense) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- HuggingFace API key
- Qdrant Cloud account
- Groq API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SalesGPT-RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
# HuggingFace API for embeddings
HF_API_KEY=your_huggingface_api_key_here

# Qdrant Vector Database
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=https://your-cluster.qdrant.io

# Groq API for LLM generation
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## Usage

### Document Ingestion
1. Navigate to the "Ingest" page
2. Upload PDF documents
3. Specify industry metadata
4. Click "Ingest" to process and index documents

### Generating Sales Content
1. Go to the "Query" page
2. Enter target company and industry
3. Select content type (pitch email, demo script, use-case brief)
4. Add focus areas or specific requirements
5. Generate personalized content

### Lead Analytics
1. Visit the "Insights" page
2. Upload CRM data (CSV format)
3. Train clustering models
4. Analyze lead segments and patterns

## Project Structure

```
SalesGPT-RAG/
├── app.py              # Streamlit web application
├── core.py             # Core RAG functionality
├── ml.py               # Machine learning models
├── embedding.py        # HuggingFace embedding utilities
├── conect.py           # Database connections
├── data/               # Data storage
│   ├── raw/           # Original documents
│   ├── processed/     # Processed data
│   └── bm25_model.pkl # BM25 search model
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

The project uses the following key libraries:

- **Streamlit**: Web application framework
- **Qdrant Client**: Vector database operations
- **Sentence Transformers**: Text embeddings via HuggingFace
- **Scikit-learn**: Machine learning algorithms
- **PyPDF2**: PDF text extraction
- **Rank BM25**: Keyword-based search
- **Requests**: HTTP API calls
- **Pandas & NumPy**: Data manipulation

## API Integrations

- **HuggingFace**: Text embeddings and model inference
- **Qdrant**: Vector database for similarity search
- **Groq**: Fast LLM inference for content generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open a GitHub issue or contact the development team.
