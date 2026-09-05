# 🪑 FurnitureGPT

> An AI-powered Furniture Recommendation & Semantic Search System built using **Retrieval-Augmented Generation (RAG)**, **FAISS**, **Product Quantization (PQ)**, **Inverted File Index (IVF)**, **Sentence Transformers**, **Groq LLM**, and **Streamlit**.

---

## ✨ Overview

FurnitureGPT helps customers discover furniture products using natural language queries.

Instead of relying on keyword matching, the system understands user intent through semantic embeddings, retrieves the most relevant products using FAISS, and generates intelligent responses using a Large Language Model.

The project also demonstrates **vector compression using Product Quantization (PQ)** and **fast approximate nearest neighbor search using IVF**, significantly reducing memory usage while maintaining high retrieval quality.

---

## 🚀 Features

- 📄 Intelligent PDF Catalog Parsing
- ✂️ Hybrid Chunking Strategy
- 🧠 Semantic Embeddings (BAAI/bge-base-en-v1.5)
- ⚡ FAISS Vector Database
- 📦 Product Quantization (PQ)
- 🗂️ Inverted File Index (IVF)
- 🤖 Retrieval-Augmented Generation (RAG)
- 💬 Groq LLM Integration
- 🪑 Furniture Recommendation Chatbot
- 🖼️ Product Image Recommendations
- 📊 Compression & Retrieval Evaluation
- 🎨 Beautiful Streamlit UI

---

## 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Framework | Streamlit |
| Embeddings | Sentence Transformers |
| Model | BAAI/bge-base-en-v1.5 |
| Vector Database | FAISS |
| Search | IVF + Product Quantization |
| LLM | Groq |
| Parsing | PyMuPDF (fitz) |
| Environment | Python, VS Code |

---

## 🏗️ Project Pipeline

```text
Furniture Catalog PDF
        │
        ▼
PDF Parsing
        │
        ▼
Hybrid Chunking
        │
        ▼
Metadata Extraction
        │
        ▼
Sentence Embeddings
        │
        ▼
FAISS Vector Database
        │
        ▼
IVF + Product Quantization
        │
        ▼
Semantic Retrieval
        │
        ▼
Groq LLM
        │
        ▼
AI Response + Product Recommendations
```

---

## 📊 Evaluation

The project evaluates:

- ✅ Compression Ratio
- ✅ Recall@K
- ✅ Retrieval Time

These metrics compare the performance of compressed embeddings against the original embeddings.

---

## 📸 Screenshots

### Home Page

> *(Add Screenshot Here)*

---

### Chat Interface

> *(Add Screenshot Here)*

---

### Product Recommendations

> *(Add Screenshot Here)*

---

### Product Details Dialog

> *(Add Screenshot Here)*

---

## 📂 Project Structure

```text
FurnitureGPT/
│
├── images/
├── data/
├── local_pdf_parser.py
├── embeddings.py
├── database.py
├── retrieval.py
├── llm.py
├── template.py
├── app.py
├── style.css
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/FurnitureGPT.git

cd FurnitureGPT

pip install -r requirements.txt

streamlit run app.py
```

---

## 🎯 Future Improvements

- Voice Assistant
- Multi-turn Memory
- Image-to-Product Search (CLIP)
- Customer Preference Learning
- Personalized Recommendations
- Cloud Deployment
- Multi-language Support

---

## 👨‍💻 Author

**Muhammad Ali**

AI Undergraduate | LLM Engineer | Information Retrieval & RAG Enthusiast

- 💼 LinkedIn: www.linkedin.com/in/muhammad-ali-698b5233a
- 📧 Email: aliqaiser1123@gmail.com

---

## ⭐ If you found this project useful, consider giving it a Star!
