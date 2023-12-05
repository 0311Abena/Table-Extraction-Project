# Table-Extraction-Project

## **Table Extraction and Question Answering Systems** 

### Overview

These projects encompass two separate systems designed for extracting tabular data from PDFs and answering questions based on the extracted tables. 
The first system uses OpenAI's language models for question answering, while the second leverages Hugging Face's transformer models, specifically designed for table-based data.

### Features

**PDF Table Extraction:** Both systems feature a table extraction module that uses Camelot to parse and extract tables from uploaded PDF documents.
**Question Answering:**
- **OpenAI System:** Integrates OpenAI's GPT-3 for robust natural language understanding and question answering.
- **Hugging Face System:** Utilizes Hugging Face's TAPAS, a transformer model particularly fine-tuned for tabular data, providing context-aware answers.
- **Interactive Web Application:** Built using Streamlit, the applications offer a friendly UI, enabling users to upload documents, view extracted tables, and interactively ask questions.
- **Security:** Implements secure management of API tokens and sensitive data through environment variables.

### Technologies

- **Python:** Primary programming language for development.
- **Streamlit:** Framework for building the interactive web app.
- **Camelot:** Library for extracting tables from PDF documents.
- **OpenAI and Hugging Face Transformers:** For powering the question-answering models.
- **PyPDF2:** For reading PDF files.
- **Pandas:** Used for handling tabular data.
- **dotenv:** For loading environment variables from .env files.

  ### Usage
  
After launching the web app, users can upload a PDF file, view the tables extracted from it, and input questions related to the data. 
The systems process these questions using their respective NLP models and return the answers, facilitating an interactive data exploration experience.
