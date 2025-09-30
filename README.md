# AI-Resume-Screener
AI-Resume-Screener helps to analyze resume based on job description

**AI Resume Screener – Powered by Gemini and RAG**
Introduction

Recruitment is a time-consuming and resource-intensive process. HR professionals often sift through hundreds or even thousands of resumes to find the right candidate for a specific job role. Traditional screening involves keyword searches, manual filtering, and subjective evaluation, which can be slow, inconsistent, and prone to human bias.
To solve these challenges, the AI Resume Screener leverages modern Large Language Models (LLMs), embeddings, and vector databases to automate resume analysis and candidate-job matching. Built with Google’s Gemini models, LangChain, and Streamlit, the system provides a smart, AI-powered platform that evaluates resumes, generates structured insights, and allows natural language querying over stored resumes.
This project combines retrieval-augmented generation (RAG) with semantic search to not only analyze individual resumes but also build a knowledge base of resumes that recruiters can query intelligently.

Objectives of AI Resume Screener
Automated Resume Analysis
Score resumes against job descriptions.
Identify skill matches, experience relevance, strengths, and weaknesses.
AI-Powered Recommendations
Generate final hiring recommendations with explanations.
Ensure consistency across candidate evaluations.
Resume Knowledge Base (Vector Store)
Store resumes as embeddings in a ChromaDB vector database.
Enable recruiters to query across all stored resumes using natural language.
Scalable & Flexible Architecture
Handle multiple file formats (PDF, DOCX, TXT).
Allow integration with any ATS (Applicant Tracking System).
System Architecture

The system is divided into three main modules:
1. Resume Processing (resume_processor.py)
This module is responsible for loading resumes, analyzing them with Gemini, and storing them in a vector database. It contains four major functions:
load_resume(file_path)
Detects the file format (PDF, DOCX, TXT) and loads the resume content using specialized loaders.
PDF → PyPDFLoader
DOCX → Docx2txtLoader
TXT → TextLoader
analyze_resume(docs, job_description)
Splits resumes into chunks using RecursiveCharacterTextSplitter and sends prompts to Gemini LLM (gemini-2.5-pro).
Each chunk is compared with the job description, and the LLM generates:

Suitability Score (out of 100)
Skills Matched
Experience Relevance
Education Evaluation
Strengths
Weaknesses
Final Recommendation
store_to_vectorstore(text_chunks)
Converts resume chunks into embeddings using GoogleGenerativeAIEmbeddings (text-embedding-004) and stores them in ChromaDB.
This enables long-term storage of resumes for semantic retrieval.

run_self_query(query)
Implements semantic search using SelfQueryRetriever, allowing recruiters to ask natural questions such as:
“Show me Python developers with AWS experience.”
“Candidates with at least 5 years of project management.”
The retriever interprets the query, searches relevant embeddings in ChromaDB, and returns matching resume chunks.
2. Application Layer (app.py)
The Streamlit web app provides a simple and interactive user interface for recruiters and hiring managers. Key features include:
Job Description Input
A text area where recruiters paste the job requirements.
Resume Upload
Upload resumes in multiple formats (PDF, DOCX, TXT).

Resume Analysis
Clicking Analyze & Store triggers:
Resume analysis by Gemini.
Structured insights generation.
Storing resume embeddings in ChromaDB.
AI Resume Summary
Displays a detailed AI-generated report including suitability score, matched skills, and recommendations. Recruiters can also download the report as a text file.

Resume Search
A text input field where recruiters can type natural queries. For example:
“Data scientists with NLP experience”
“Candidates with an MBA degree and cloud certifications”
The AI retrieves the most relevant resume chunks and displays them in ranked order.
3. Configuration and Dependencies.env File
Stores the GOOGLE_API_KEY required to authenticate with Gemini and embeddings APIs.

Dependencies
Installed via pip:

pip install google-generativeai langchain-google-genai langchain-community python-dotenv
pip install streamlit chromadb pypdf docx2txt


Run Command
streamlit run app.py
Key Technologies Used
Google Gemini (LLMs & Embeddings)
gemini-2.5-pro: Used for resume analysis and generating structured insights.
models/text-embedding-004: Used to convert resumes into high-dimensional embeddings for semantic similarity.
LangChain Framework
Provides tools for chaining prompts, managing document loaders, and enabling retrievers.
SelfQueryRetriever allows natural language queries over stored resumes.
ChromaDB (Vector Store)
Stores embeddings of resume chunks.
Supports fast similarity search and retrieval.
Streamlit
Frontend framework for building an interactive web application.
Simplifies deployment and user experience.
File Loaders
PyPDFLoader for PDFs.
Docx2txtLoader for Word documents.

TextLoader for plain text resumes.
Workflow
Upload Resume & Paste Job Description
Recruiter uploads a resume and enters job requirements.
AI Analysis
Resume content is split into chunks.
Gemini compares each chunk with the job description.
AI generates a structured evaluation (skills, experience, suitability score, strengths, weaknesses).
Storing in Vector Database

Resume text chunks are embedded using Google embeddings.
Embeddings are stored in ChromaDB with metadata.
Smart Resume Search
Recruiter types a natural query.
SelfQueryRetriever interprets the query.
Relevant resumes are retrieved and displayed.

**Benefits**
Time Efficiency
Automates the repetitive process of resume screening.
Consistency
Every resume is evaluated against the same metrics, reducing bias.
Smart Search Capability
Enables recruiters to find specific candidates instantly.
Scalability
Works with hundreds or thousands of resumes stored in the vector database.
Explainability
Provides detailed breakdowns of why a candidate is suitable or not.
