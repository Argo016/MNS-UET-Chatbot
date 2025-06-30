import os
import warnings
import json
import numpy as np
import pandas as pd
import nltk
import fitz  # PyMuPDF
import pdfplumber  # For table extraction
import faiss
import streamlit as st
import hashlib
import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
from datasets import Dataset
import asyncio
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# New imports for web scraping
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    st.warning("RAGAS not installed. Install with: pip install ragas")

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="MNS-UET Chatbot"
    page_icon="📄"
    initial_sidebar_state="collapsed"
)

# Set GEMINI API key
api_key = os.getenv("GEMINI_API_KEY")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Extract text from PDFs with improved coverage and fallback
def extract_text_from_pdfs(pdf_paths: list) -> list:
    chunks = []

    for pdf_path in pdf_paths:
        try:
            if not os.path.exists(pdf_path):
                st.error(f"File not found: {pdf_path}")
                continue

            doc = fitz.open(pdf_path)
            doc_name = os.path.basename(pdf_path)
            st.info(f"Processing {doc_name}: {len(doc)} pages")

            with pdfplumber.open(pdf_path) as pdf:
                total_lines_processed = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")

                    if not text.strip():
                        st.warning(f"Skipping empty page {page_num + 1} in {doc_name}")
                        continue

                    # Extract tables with pdfplumber
                    plumber_page = pdf.pages[page_num]
                    tables = plumber_page.extract_tables()
                    table_idx = 0
                    if tables:
                        for table in tables:
                            if table and any(row for row in table if any(cell.strip() for cell in row)):
                                table_text = []
                                for row in table:
                                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                                    if any(cleaned_row):
                                        table_text.append(" | ".join(cleaned_row))
                                if table_text:
                                    table_content = "\n".join([f"• {row}" for row in table_text])
                                    chunks.append({
                                        "text": table_content,
                                        "doc_name": doc_name,
                                        "page_num": page_num + 1,
                                        "paragraph": 0,
                                        "chunk_id": f"{doc_name}_p{page_num + 1}_table{table_idx}"
                                    })
                                    table_idx += 1

                    # Extract text lines with section detection and fallback
                    lines = text.split('\n')
                    current_section = None
                    section_content = []
                    lines_processed = 0

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        lines_processed += 1
                        total_lines_processed += 1
                        # Detect section headers
                        if re.match(r'^(Faculty|Department)\s+of\s+[A-Za-z\s]+$', line):
                            if section_content:
                                chunks.append({
                                    "text": f"{current_section}\n\n" + "\n".join(section_content) if current_section else "\n".join(section_content),
                                    "doc_name": doc_name,
                                    "page_num": page_num + 1,
                                    "paragraph": len(chunks) + 1,
                                    "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                                })
                                section_content = []
                            current_section = line
                        elif line and (line.startswith(('Engr.', 'Dr.', 'Prof.', 'Lecturer', 'MSc.', 'Area of Interest')) or re.search(r'\b[A-Za-z]+\s+[A-Za-z]+\b', line)):
                            section_content.append(line)
                        elif section_content:
                            section_content.append(line)

                    if section_content:
                        chunks.append({
                            "text": f"{current_section}\n\n" + "\n".join(section_content) if current_section else "\n".join(section_content),
                            "doc_name": doc_name,
                            "page_num": page_num + 1,
                            "paragraph": len(chunks) + 1,
                            "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                        })
                    # Fallback: Extract all remaining lines as individual chunks
                    if not chunks or chunks[-1]["page_num"] != page_num + 1:
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 10:  # Avoid empty or short lines
                                chunks.append({
                                    "text": line,
                                    "doc_name": doc_name,
                                    "page_num": page_num + 1,
                                    "paragraph": len(chunks) + 1,
                                    "chunk_id": f"{doc_name}_p{page_num + 1}_par{len(chunks) + 1}"
                                })
                    st.write(f"Processed {lines_processed} lines on page {page_num + 1}")

            st.success(f"Extracted {len(chunks)} chunks from {doc_name} (Total lines processed: {total_lines_processed})")
            doc.close()
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {str(e)}")

    st.info(f"Total chunks extracted: {len(chunks)}")
    return chunks

# New function for web scraping
def scrape_website(base_url: str, max_pages: int = 20, delay: float = 1.0) -> list:
    web_chunks = []
    visited_urls = set()
    urls_to_visit = [base_url]
    domain = urlparse(base_url).netloc
    page_counter = 0

    st.info(f"Starting web scraping from {base_url} (max {max_pages} pages, {delay}s delay)...")

    while urls_to_visit and page_counter < max_pages:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        st.write(f"Scraping: {current_url}")
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract main text content
            page_text = ""
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span']):
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    page_text += text + "\n"

            if page_text.strip():
                # Split text into chunks, similar to PDF processing if needed, or keep as one chunk per page
                # For simplicity, let's treat the whole page as one chunk for now
                web_chunks.append({
                    "text": page_text,
                    "doc_name": f"web_{domain}",
                    "page_num": page_counter + 1, # Simulate page numbers for web content
                    "paragraph": 1,
                    "chunk_id": f"web_{domain}_page{page_counter + 1}"
                })
                page_counter += 1

            visited_urls.add(current_url)

            # Find and queue new URLs on the same domain
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                full_url = urljoin(base_url, href)
                parsed_full_url = urlparse(full_url)

                if parsed_full_url.netloc == domain and full_url not in visited_urls and full_url not in urls_to_visit:
                    # Basic check to avoid common non-content links like anchors, mailto, etc.
                    if not full_url.endswith(('.pdf', '.jpg', '.png', '.zip', '.rar')) and not full_url.startswith('mailto:'):
                        urls_to_visit.append(full_url)

            time.sleep(delay) # Respectful delay

        except requests.exceptions.RequestException as e:
            st.warning(f"Could not retrieve {current_url}: {e}")
        except Exception as e:
            st.error(f"Error processing {current_url}: {e}")

    st.success(f"Finished web scraping. Extracted {len(web_chunks)} chunks.")
    return web_chunks

# 2. Create FAISS index
def create_vector_store(chunks: list) -> tuple:
    if not chunks:
        st.error("No chunks to embed. Check PDF and web extraction.")
        return None, None, []

    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    texts = [chunk["text"] for chunk in chunks]

    progress_text = st.empty()
    progress_bar = st.progress(0)
    batch_size = 32
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_number = i // batch_size + 1
        progress_text.text(f"Embedding batch {batch_number}/{total_batches}...")
        progress_bar.progress(batch_number / total_batches)
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    progress_text.text("Creating FAISS index...")
    progress_bar.progress(0.9)

    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    progress_text.empty()
    progress_bar.empty()
    st.success(f"Successfully created vector store with {len(chunks)} chunks")
    return index, model, chunks

# 3. Retrieve chunks with chunk_id-based indexing
def retrieve_chunks(query: str, index, model, chunks: list, k: int = 15) -> list:
    query_emb = model.encode([query])[0]
    faiss.normalize_L2(np.array([query_emb]))
    distances, indices = index.search(np.array([query_emb]), k)

    initial_results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["similarity"] = float(distances[0][i])
            initial_results.append(chunk)

    expanded_results = []
    seen_ids = set()
    for result in initial_results:
        if result["chunk_id"] not in seen_ids:
            expanded_results.append(result)
            seen_ids.add(result["chunk_id"])
            # Find the original index using chunk_id
            for idx, chunk in enumerate(chunks):
                if chunk["chunk_id"] == result["chunk_id"]:
                    current_idx = idx
                    break
            page_num = result["page_num"]
            doc_name = result["doc_name"]
            
            # Context expansion: look for chunks from the same document and nearby pages
            for offset in range(-2, 3):  # Look 2 chunks before and after
                new_idx = current_idx + offset
                if 0 <= new_idx < len(chunks):
                    nearby_chunk = chunks[new_idx]
                    # Ensure same document and relatively close pages for context
                    if (nearby_chunk["chunk_id"] not in seen_ids and
                        nearby_chunk["doc_name"] == doc_name and
                        abs(nearby_chunk["page_num"] - page_num) <= 1): # Adjust page proximity for context
                        nearby_chunk = nearby_chunk.copy()
                        nearby_chunk["similarity"] = result["similarity"]  # Inherit similarity
                        expanded_results.append(nearby_chunk)
                        seen_ids.add(nearby_chunk["chunk_id"])

    return expanded_results

# 4. Generate answer with detailed synthesis
def generate_answer(query: str, retrieved_chunks: list, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    doc_pages = {}
    for chunk in retrieved_chunks:
        doc_key = chunk['doc_name']
        page_key = chunk['page_num']
        if doc_key not in doc_pages:
            doc_pages[doc_key] = {}
        if page_key not in doc_pages[doc_key]:
            doc_pages[doc_key][page_key] = []
        doc_pages[doc_key][page_key].append(chunk)

    context_items = []
    for doc_name, pages in doc_pages.items():
        doc_content = f"# Document: {doc_name}\n\n"
        for page_num, page_chunks in sorted(pages.items()):
            page_chunks.sort(key=lambda x: (x['paragraph'], x['chunk_id']))
            page_text = "\n".join([chunk['text'] for chunk in page_chunks])
            doc_content += f"## Page {page_num}:\n{page_text}\n\n"
        context_items.append(doc_content)

    context = "\n".join(context_items)

    prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents and web content.

Document Context:
{context}

User Question: {query}

Instructions:
1. Start with a clear, concise conclusion that directly answers the question using information from ALL relevant parts of the documents and web content.
2. For queries about a group of people (e.g., "faculty of electrical"), list all individuals mentioned in the relevant section, including their names, titles, qualifications, and areas of interest if available.
3. For queries about a specific person (e.g., "who is Engr. Dr. Muhammad Shahzad"), provide all available details about them, including their role, faculty/department affiliation, qualifications, and areas of interest.
4. After the conclusion, provide additional details or context if necessary to support the answer.
5. At the end, list the sources used in a section called 'Sources', citing the document name and page number (e.g., "Sources: [prospectus.pdf, Page 5]", "Sources: [web_example.com, Page 1]").
6. Do NOT include inline citations in the main answer (e.g., avoid "[doc, Page 5]" within the text).
7. If the documents don't contain the answer, say: "I don't have enough information to answer this question."
8. Ensure all relevant individuals or items are included, even if spread across multiple pages or sections.
9. Keep the answer organized, readable, and focused on the user's query.

Your Answer:"""

    try:
        response = model.generate_content(prompt)
        referenced_sources = []
        response_text = response.text
        sources_section = ""
        if "Sources:" in response_text:
            sources_section = response_text.split("Sources:")[1].strip()
            response_text = response_text.split("Sources:")[0].strip()

        for chunk in retrieved_chunks:
            doc = chunk['doc_name']
            page = chunk['page_num']
            source_key = f"{doc}-p{page}"
            # Adjust pattern for web sources which might have page numbers like 'Page 1'
            source_pattern = f"[{doc}, Page {page}]"
            alt_pattern = f"[{doc}, page {page}]" # Lowercase 'page'
            
            # Check if the source is explicitly mentioned in the generated sources section
            if source_pattern in sources_section or alt_pattern in sources_section:
                if not any(s["document"] == doc and s["page"] == page for s in referenced_sources):
                    referenced_sources.append({
                        "document": doc,
                        "page": page,
                        "text": chunk['text'],
                        "similarity": chunk.get('similarity', 0)
                    })

        return {
            "answer": response_text,
            "sources": referenced_sources,
            "context": context,
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        error_message = f"Error: Failed to generate answer ({str(e)})."
        return {
            "answer": error_message,
            "sources": [],
            "context": "",
            "retrieved_chunks": []
        }

# RAGAS Evaluation Functions (no changes needed here, as it works on the
# 'chunks' and 'responses' which are agnostic to source type)
def create_evaluation_dataset(qa_pairs: list) -> Dataset:
    """Create a dataset for RAGAS evaluation"""
    if not qa_pairs:
        return None

    data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    for qa in qa_pairs:
        data['question'].append(qa['question'])
        data['answer'].append(qa['answer'])
        data['contexts'].append(qa['contexts'])
        data['ground_truth'].append(qa.get('ground_truth', qa['answer']))  # Use answer as ground truth if not provided

    return Dataset.from_dict(data)

def evaluate_with_ragas(dataset: Dataset) -> dict:
    """Evaluate the RAG system using RAGAS metrics with Gemini"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Configure Gemini LLM for RAGAS
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

        # Configure Gemini embeddings for RAGAS
        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

        # Configure metrics to use Gemini
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ]

        # Set the LLM and embeddings for each metric
        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = ragas_embeddings

        # Run evaluation with configured metrics
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        return result
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def generate_sample_questions():
    """Generate sample questions for evaluation"""
    sample_questions = [
        {
            "question": "Who are the faculty members in the Electrical Engineering department?",
            "ground_truth": "List of electrical engineering faculty with their qualifications and areas of interest"
        },
        {
            "question": "What are the admission requirements for undergraduate programs?",
            "ground_truth": "Specific admission requirements including academic qualifications and entrance exams"
        },
        {
            "question": "What is the fee structure for different programs?",
            "ground_truth": "Detailed fee structure for undergraduate and graduate programs"
        },
        {
            "question": "What are the available research areas in Computer Science?",
            "ground_truth": "List of research areas and specializations in Computer Science department"
        },
        {
            "question": "Who is the Dean of Engineering Faculty?",
            "ground_truth": "Name and details of the Dean of Engineering Faculty"
        }
    ]
    return sample_questions

# Render source references
def render_source_references(sources, expander_key_prefix="source_expander_"):
    if not sources:
        st.info("No specific sources were referenced in this answer.")
        return

    st.markdown("### Source References")
    doc_sources = {}
    for source in sources:
        doc = source["document"]
        if doc not in doc_sources:
            doc_sources[doc] = []
        doc_sources[doc].append(source)

    if len(doc_sources) > 0:
        doc_tabs = st.tabs(list(doc_sources.keys()))
        for i, (doc, tab) in enumerate(zip(doc_sources.keys(), doc_tabs)):
            with tab:
                pages = sorted(doc_sources[doc], key=lambda x: x["page"])
                # Use st.expander to collapse sources by default
                # The expanded state is stored in session_state for persistence
                if f"{expander_key_prefix}{doc}" not in st.session_state:
                    st.session_state[f"{expander_key_prefix}{doc}"] = False # Collapsed by default

                with st.expander(f"Show references from {doc}", expanded=st.session_state[f"{expander_key_prefix}{doc}"]):
                    # Update session state when expander is interacted with
                    st.session_state[f"{expander_key_prefix}{doc}"] = True # If opened, keep it open

                    for j, page_info in enumerate(pages):
                        st.markdown(f"**Page {page_info['page']} (Relevance: {page_info['similarity']:.2f})**")
                        st.text(page_info['text'])
                        # MODIFIED LINE FOR UNIQUE KEY GENERATION
                        unique_key = f"{expander_key_prefix}_doc_{i}_page_{j}_highlight_button_{hashlib.md5(page_info['text'].encode()).hexdigest()}"
                        if st.button(f"Highlight this reference", key=unique_key):
                            st.session_state.highlight_source = f"{doc}-p{page_info['page']}"
                            st.rerun()
                        st.divider()

def render_ragas_evaluation():
    """Render RAGAS evaluation interface"""
    st.header("🔍 RAGAS Evaluation")

    if not RAGAS_AVAILABLE:
        st.error("RAGAS is not installed. Please install it using: `pip install ragas`")
        return

    # Initialize evaluation data in session state
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = []

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Evaluation Dataset")

        # Option to use sample questions
        if st.button("Load Sample Questions"):
            sample_questions = generate_sample_questions()
            st.session_state.evaluation_data.extend(sample_questions)
            st.success("Sample questions loaded!")

        # Manual question entry
        with st.expander("Add Custom Question"):
            question = st.text_input("Question (for evaluation):")
            ground_truth = st.text_area("Ground Truth Answer (for evaluation):")

            if st.button("Add Question to Evaluation") and question and ground_truth:
                # Generate answer for this question using current RAG setup
                if "index" in st.session_state and "model" in st.session_state and "chunks" in st.session_state:
                    with st.spinner("Generating answer for evaluation..."):
                        retrieved_chunks = retrieve_chunks(
                            question,
                            st.session_state.index,
                            st.session_state.model,
                            st.session_state.chunks
                        )
                        response = generate_answer(question, retrieved_chunks, api_key)

                        eval_item = {
                            "question": question,
                            "answer": response["answer"],
                            "contexts": [chunk["text"] for chunk in retrieved_chunks],
                            "ground_truth": ground_truth,
                            "timestamp": datetime.now().isoformat()
                        }

                        st.session_state.evaluation_data.append(eval_item)
                        st.success("Question added to evaluation dataset!")
                else:
                    st.warning("Please process documents/scrape website first before adding questions for evaluation.")


        # Use chat history for evaluation
        if st.button("Use Chat History for Evaluation"):
            if st.session_state.messages:
                eval_data = []
                for i in range(0, len(st.session_state.messages), 2):
                    if i + 1 < len(st.session_state.messages):
                        user_msg = st.session_state.messages[i]
                        assistant_msg = st.session_state.messages[i + 1]

                        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                            # Get contexts from the assistant message
                            contexts = []
                            if "retrieved_chunks" in assistant_msg:
                                contexts = [chunk["text"] for chunk in assistant_msg["retrieved_chunks"]]

                            eval_item = {
                                "question": user_msg["content"],
                                "answer": assistant_msg.get("answer", ""),
                                "contexts": contexts,
                                "ground_truth": assistant_msg.get("answer", ""),  # Use answer as ground truth if no ground_truth provided
                                "timestamp": datetime.now().isoformat()
                            }
                            eval_data.append(eval_item)

                st.session_state.evaluation_data.extend(eval_data)
                st.success(f"Added {len(eval_data)} Q&A pairs from chat history")
            else:
                st.info("No chat history to use for evaluation.")

        # Display current evaluation dataset
        if st.session_state.evaluation_data:
            st.subheader(f"Current Dataset ({len(st.session_state.evaluation_data)} items)")

            for i, item in enumerate(st.session_state.evaluation_data):
                with st.expander(f"Q{i+1}: {item['question'][:50]}..."):
                    st.write("**Question:**", item['question'])
                    st.write("**Generated Answer:**", item['answer'][:200] + "...")
                    st.write("**Ground Truth:**", item['ground_truth'][:200] + "...")
                    st.write("**Contexts:**", f"{len(item['contexts'])} chunks")
                    if st.button(f"Remove Q{i+1}", key=f"remove_{i}"):
                        st.session_state.evaluation_data.pop(i)
                        st.rerun()

    with col2:
        st.subheader("Run Evaluation")

        if st.session_state.evaluation_data:
            if st.button("🚀 Run RAGAS Evaluation", type="primary"):
                try:
                    with st.spinner("Running RAGAS evaluation..."):
                        # Create dataset
                        dataset = create_evaluation_dataset(st.session_state.evaluation_data)

                        if dataset:
                            # Run evaluation
                            results = evaluate_with_ragas(dataset)
                            st.session_state.evaluation_results = results
                            st.success("Evaluation completed!")
                        else:
                            st.error("Failed to create evaluation dataset. Dataset might be empty.")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.session_state.evaluation_results = {"error": str(e)}

        if st.button("Clear Evaluation Data"):
            st.session_state.evaluation_data = []
            st.session_state.evaluation_results = None
            st.success("Evaluation data cleared!")

    # Display evaluation results
    if st.session_state.evaluation_results:
        st.subheader("📊 Evaluation Results")

        results = st.session_state.evaluation_results

        # Check if results is an error dictionary
        if isinstance(results, dict) and "error" in results:
            st.error(results["error"])
        else:
            try:
                # Display metrics
                col1, col2, col3 = st.columns(3)

                metrics_to_display = [
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevancy", "Answer Relevancy"),
                    ("context_precision", "Context Precision"),
                    ("context_recall", "Context Recall"),
                    ("answer_similarity", "Answer Similarity"),
                    ("answer_correctness", "Answer Correctness")
                ]

                # Handle different result formats
                if hasattr(results, 'to_pandas'):
                    # If results can be converted to pandas DataFrame
                    df = results.to_pandas()

                    # Display average metrics
                    for i, (metric_key, metric_name) in enumerate(metrics_to_display):
                        col = [col1, col2, col3][i % 3]
                        with col:
                            if metric_key in df.columns:
                                value = df[metric_key].mean()
                                st.metric(metric_name, f"{value:.3f}")

                    # Show detailed results
                    with st.expander("Detailed Results"):
                        st.dataframe(df)

                elif hasattr(results, '__dict__') or hasattr(results, 'keys'):
                    # If results is a dictionary-like object
                    for i, (metric_key, metric_name) in enumerate(metrics_to_display):
                        col = [col1, col2, col3][i % 3]
                        with col:
                            try:
                                if hasattr(results, metric_key):
                                    value = getattr(results, metric_key)
                                elif hasattr(results, '__getitem__'):
                                    value = results[metric_key]
                                else:
                                    continue

                                if isinstance(value, (int, float)):
                                    st.metric(metric_name, f"{value:.3f}")
                                elif hasattr(value, 'mean'):
                                    st.metric(metric_name, f"{value.mean():.3f}")
                                else:
                                    st.metric(metric_name, str(value))
                            except (KeyError, AttributeError):
                                continue

                    # Show detailed results
                    with st.expander("Detailed Results"):
                        st.write(results)

                else:
                    # Fallback for unknown result format
                    st.write("Results:")
                    st.write(results)

                # Export results
                if st.button("📄 Export Results"):
                    try:
                        if hasattr(results, 'to_pandas'):
                            results_data = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset_size": len(st.session_state.evaluation_data),
                                "metrics": results.to_pandas().to_dict()
                            }
                        else:
                            results_data = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset_size": len(st.session_state.evaluation_data),
                                "metrics": str(results)
                            }

                        st.download_button(
                            label="Download Results JSON",
                            data=json.dumps(results_data, indent=2, default=str),
                            file_name=f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error exporting results: {str(e)}")

            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                with st.expander("Raw Results"):
                    st.write(results)

# Render processed content for UI
def render_processed_content(chunks):
    if not chunks:
        st.info("No content has been processed yet.")
        return

    st.markdown("### Processed Content Overview")
    
    # Group chunks by document
    docs_content = {}
    for chunk in chunks:
        doc_name = chunk['doc_name']
        if doc_name not in docs_content:
            docs_content[doc_name] = []
        docs_content[doc_name].append(chunk)

    if len(docs_content) > 0:
        doc_tabs = st.tabs(list(docs_content.keys()))
        for i, (doc_name, doc_chunks) in enumerate(zip(docs_content.keys(), doc_tabs)):
            with doc_chunks:
                # Sort chunks by page number and then paragraph/chunk_id for ordered display
                sorted_chunks = sorted(doc_chunks, key=lambda x: (x['page_num'], x['paragraph'] if 'paragraph' in x else 0, x['chunk_id']))
                
                current_page = None
                for chunk in sorted_chunks:
                    if chunk['page_num'] != current_page:
                        st.subheader(f"Page {chunk['page_num']}")
                        current_page = chunk['page_num']
                    st.write(f"Chunk ID: {chunk['chunk_id']}")
                    st.code(chunk['text'], language="text") # Display chunk text in a code block for readability
                    st.markdown("---")


# Streamlit app
def main():
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "highlight_source" not in st.session_state:
        st.session_state.highlight_source = None
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = {}
    if "processed_sources" not in st.session_state:
        st.session_state.processed_sources = {"pdfs": [], "web": []}
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    # Persistent storage flags
    INDEX_FILE = "index.faiss"
    CHUNKS_FILE = "chunks.json"
    PROCESSED_SOURCES_FILE = "processed_sources.json"


    # Sidebar
    with st.sidebar:
        st.title(" MNS-UET Chatbot")

        # Navigation
        tab_selection = st.radio(
            "Choose Mode:",
            ["💬 Chat", "🔍 RAGAS Evaluation"],
            key="main_tabs"
        )

        st.divider()

        st.subheader("Document Sources")
        st.write("This chatbot can answer questions based on PDF documents and scraped web content.")

        pdf_paths = ["rules.pdf", "prospectus.pdf"]
        missing_pdf_files = [f for f in pdf_paths if not os.path.exists(f)]
        if missing_pdf_files:
            st.warning(f"Missing PDF files: {', '.join(missing_pdf_files)}. Please place these PDF files in the root directory.")
        else:
            st.success("All default PDF files found.")

        st.markdown("---")
        st.subheader("Web Scraping")
        # Hardcoded website URL for scraping
        fixed_web_url = "https://mnsuet.edu.pk/" # You can change this to any desired fixed URL
        st.info(f"Fixed website to scrape: {fixed_web_url}")

        max_web_pages = st.slider("Max pages to scrape from URL", min_value=1, max_value=100, value=200)
        scraping_delay = st.slider("Delay between page requests (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)


        st.divider()
        st.subheader("Parameters")
        k_value = st.slider("Number of chunks to retrieve", min_value=3, max_value=50, value=30)

        if st.button("Process Documents & Scrape Web", type="primary"):
            # Clear previous session state related to processed data
            st.session_state.pop("index", None)
            st.session_state.pop("model", None)
            st.session_state.pop("chunks", None)
            st.session_state.processed_sources = {"pdfs": [], "web": []} # Reset processed sources
            st.session_state.processing_done = False # Mark for reprocessing
            st.success("Documents will be reprocessed and web content scraped!")
            st.rerun() # Rerun to trigger reprocessing

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.highlight_source = None
            st.session_state.show_sources = {}
            st.success("Chat history cleared!")

        if "chunks" in st.session_state:
            with st.expander("Dataset Stats"):
                chunks = st.session_state.chunks
                st.write(f"Total chunks: {len(chunks)}")
                docs = {}
                for chunk in chunks:
                    doc = chunk["doc_name"]
                    if doc not in docs:
                        docs[doc] = set()
                    docs[doc].add(chunk["page_num"])
                for doc, pages in docs.items():
                    source_type = "Web" if doc.startswith("web_") else "PDF"
                    st.write(f"• {doc} ({source_type}): {len(pages)} pages")
                
                # Add a button to view raw processed content
                if st.button("View All Processed Content", key="view_all_content_btn"):
                    st.session_state.view_all_processed_content = True
                    st.rerun()


    # Main content area
    if tab_selection == "🔍 RAGAS Evaluation":
        render_ragas_evaluation()
        return

    # Chat interface
    st.title("What can I do for you today?")

    # Display all processed content if requested
    if st.session_state.get("view_all_processed_content", False) and "chunks" in st.session_state:
        render_processed_content(st.session_state.chunks)
        if st.button("Hide Processed Content", key="hide_all_content_btn"):
            st.session_state.view_all_processed_content = False
            st.rerun()
        st.markdown("---") # Add a separator after the content view
        
    # Persistent storage and initial processing logic
    if not st.session_state.processing_done:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Check for existing index and chunks
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(PROCESSED_SOURCES_FILE):
            status_text.text("Loading existing index and data...")
            progress_bar.progress(30)
            try:
                index = faiss.read_index(INDEX_FILE)
                progress_bar.progress(60)
                with open(CHUNKS_FILE, "r") as f:
                    chunks = json.load(f)
                progress_bar.progress(80)
                with open(PROCESSED_SOURCES_FILE, "r") as f:
                    st.session_state.processed_sources = json.load(f)
                progress_bar.progress(90)
                model = SentenceTransformer('BAAI/bge-small-en-v1.5') # Model needs to be loaded regardless
                progress_bar.progress(100)
                st.session_state.index = index
                st.session_state.model = model
                st.session_state.chunks = chunks
                st.session_state.processing_done = True
                st.success(f"Loaded existing index with {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Error loading saved data: {e}. Reprocessing.")
                st.session_state.pop("index", None)
                st.session_state.pop("model", None)
                st.session_state.pop("chunks", None)
                st.session_state.processed_sources = {"pdfs": [], "web": []}
                st.session_state.processing_done = False # Revert to False to trigger reprocessing below

        # If not loaded, or if reprocessing requested, or if error occurred
        if not st.session_state.processing_done:
            all_chunks = []
            
            # Process PDFs
            status_text.text("Processing PDF documents...")
            progress_bar.progress(10)
            pdf_chunks = extract_text_from_pdfs(pdf_paths)
            all_chunks.extend(pdf_chunks)
            st.session_state.processed_sources["pdfs"] = pdf_paths # Store which PDFs were processed

            # Process Web content using the fixed URL
            # No need for a conditional check on web_url's validity as it's hardcoded
            status_text.text(f"Scraping web content from {fixed_web_url}...")
            progress_bar.progress(30)
            web_chunks = scrape_website(fixed_web_url, max_pages=max_web_pages, delay=scraping_delay)
            all_chunks.extend(web_chunks)
            st.session_state.processed_sources["web"] = [fixed_web_url] # Store the scraped URL


            progress_bar.progress(50)
            status_text.text("Creating vector embeddings...")
            index, model, chunks = create_vector_store(all_chunks)
            st.session_state.index = index
            st.session_state.model = model
            st.session_state.chunks = chunks

            # Save the new index and chunks for persistence
            status_text.text("Saving index and chunks for persistence...")
            faiss.write_index(index, INDEX_FILE)
            with open(CHUNKS_FILE, "w") as f:
                json.dump(chunks, f)
            with open(PROCESSED_SOURCES_FILE, "w") as f:
                json.dump(st.session_state.processed_sources, f)
            progress_bar.progress(100)
            
            st.session_state.processing_done = True

        status_text.empty()
        progress_bar.empty()
        if st.session_state.processing_done:
            st.success("All content processed and index created/loaded!")
    else:
        index = st.session_state.index
        model = st.session_state.model
        chunks = st.session_state.chunks

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "answer" in message:
                answer_text = message["answer"]
                if st.session_state.highlight_source and "sources" in message:
                    highlight_key = st.session_state.highlight_source
                    doc_name, page_str = highlight_key.split("-p")
                    page_num = int(page_str)
                    patterns = [
                        f"[{doc_name}, Page {page_num}]",
                        f"[{doc_name}, page {page_num}]"
                    ]
                    highlighted_text = answer_text
                    for pattern in patterns:
                        if pattern in highlighted_text:
                            highlighted_text = highlighted_text.replace(
                                pattern,
                                f"**:red[{pattern}]**"
                            )
                    st.markdown(highlighted_text)
                    st.session_state.highlight_source = None
                else:
                    if len(answer_text) > 1500:
                        st.markdown(answer_text[:1500] + "...")
                        with st.expander("Show full answer"):
                            st.markdown(answer_text)
                    else:
                        st.markdown(answer_text)

                if "sources" in message and message["sources"]:
                    # No longer using a button to toggle, directly render with expander
                    render_source_references(message["sources"], expander_key_prefix=f"msg_{i}_source_expander_")
            else:
                st.markdown(message["content"])

    # Chat input
    if query := st.chat_input("Ask a question about the documents and web content"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve_chunks(query, index, model, chunks, k=k_value)
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("Please set GEMINI_API_KEY in the environment variables.")
                    answer = "Error: API key not found."
                    sources = []
                    context = ""
                else:
                    response = generate_answer(query, retrieved_chunks, api_key)
                    answer = response["answer"]
                    sources = response["sources"]
                    context = response.get("context", "")

                if len(answer) > 1500:
                    st.markdown(answer[:1500] + "...")
                    with st.expander("Show full answer"):
                        st.markdown(answer)
                else:
                    st.markdown(answer)

                if sources:
                    st.markdown("---")
                    # Pass a unique key prefix to render_source_references
                    render_source_references(sources, expander_key_prefix=f"chat_source_expander_{len(st.session_state.messages)}")

        # Store the complete response for RAGAS evaluation
        st.session_state.messages.append({
            "role": "assistant",
            "answer": answer,
            "sources": sources,
            "context": context,
            "retrieved_chunks": retrieved_chunks
        })

if __name__ == "__main__":
    main()