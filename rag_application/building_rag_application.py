#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
"""RAG demo that ingests a medical FAQ corpus and serves a Retrieval-Augmented chatbot."""

from __future__ import annotations

import os
import traceback
import warnings
from pathlib import Path
from typing import Dict, List
import json

import gdown
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI as LangchainOpenAI
from openai import OpenAI

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATASET_ID = "1727UCcg_Nxn9Nbj_LiX8pvGEpo591oxc"
DATASET_PATH = BASE_DIR / "input" / "ai-medical-chatbot.txt"
FAISS_DIR = BASE_DIR / "faiss_doc_idx"
PROMPT_TEMPLATE = (
    "You are a medical assistant chatbot helping answer patient questions based only on the provided context.\n"
    "Do not guess or provide inaccurate information. If the answer is not found in the context, say you don’t know.\n"
    "You will answer the question based on the context - {context}.\n"
    "Question: {question}\n"
    "Answer:"
)


def init_openai_client() -> OpenAI:
    """Load environment variables and return an OpenAI client."""
    load_dotenv(BASE_DIR / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to rag_application/.env before running the script.")
    return OpenAI()


def ensure_dataset() -> str:
    """Download the dataset if necessary and return the raw text."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        print("Downloading dataset...")
        gdown.download(id=DATASET_ID, output=str(DATASET_PATH), quiet=False)
    return DATASET_PATH.read_text()


def preprocess_sections(raw_text: str) -> List[str]:
    """Clean raw sections and return a list of normalized blocks."""
    sections = raw_text.replace("\n\n", "\n").split("---")
    cleaned: List[str] = []
    for idx, section in enumerate(sections):
        text = section
        if idx == 4:
            text = text.replace("\n**", "\n###")
        cleaned.append(text.replace("**", ""))
    return cleaned


def build_qa_pairs(sections: List[str]) -> Dict[str, str]:
    """Convert cleaned sections into a dictionary of question-answer pairs."""
    qa_pairs: Dict[str, str] = {}
    for section in sections:
        topics = section.split("\n###")
        for topic in topics[1:]:
            lines = [line.strip() for line in topic.split("\n") if line.strip()]
            if not lines:
                continue
            question, *answer_lines = lines
            qa_pairs[question] = " ".join(answer_lines)
    return qa_pairs


def build_corpus(qa_pairs: Dict[str, str]) -> str:
    """Concatenate QA pairs into a single corpus string."""
    return "\n".join(f"{question} {answer}" for question, answer in qa_pairs.items())


def write_json(path: Path, data) -> None:
    """Persist Python data structures as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def chunk_corpus(corpus: str) -> List[str]:
    """Split the corpus into overlapping chunks for embedding."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=128, length_function=len)
    return splitter.split_text(corpus)


def build_vector_store(chunks: List[str]) -> FAISS:
    """Embed the chunks, persist them locally, and return a FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(str(FAISS_DIR))
    return vector_store


def run_sample_queries(vector_store: FAISS) -> None:
    """Execute a couple of sample similarity searches for sanity checks."""
    sample_questions = [
        "What are the main differences between acute and chronic medical conditions?",
        "I have pain in arm, should I consult multiple doctors?",
    ]
    for question in sample_questions:
        print(f"\nTop chunks for: {question}\n{'-' * 40}")
        docs = vector_store.similarity_search(question)
        for doc in docs:
            print(doc.page_content)


def create_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """Create a RetrievalQA chain that uses the stored embeddings."""
    llm = LangchainOpenAI()
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="similarity", k=4),
        chain_type_kwargs={"prompt": prompt},
    )


def launch_chat_interface(qa_chain: RetrievalQA) -> None:
    """Spin up a Gradio chat interface backed by the QA chain."""
    DEBUG = True

    def predict(message: str, history: List[List[str]]):
        try:
            if DEBUG:
                print(f"\nReceived message: {message}")
            result = qa_chain({"query": message})
            answer = result["result"]
            if DEBUG:
                print(f"Retrieved Answer:\n{answer}")
            return answer
        except Exception:
            error_trace = traceback.format_exc()
            if DEBUG:
                print(f"Exception occurred:\n{error_trace}")
            return "An internal error occurred. Please try again later."

    gr.ChatInterface(
        fn=predict,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(
            placeholder="Ask me a question related to Healthcare and Medical Services",
            container=False,
            scale=7,
        ),
        title="DocumentQABot",
        theme="soft",
        examples=[
            "What are the main differences between acute and chronic medical conditions?",
            "What does mild concentric LV hypertrophy mean?",
        ],
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch(share=True)


def main() -> None:
    """Run the end-to-end pipeline: ingest, index, test, and deploy."""
    init_openai_client()
    raw_text = ensure_dataset()
    sections = preprocess_sections(raw_text)
    write_json(OUTPUT_DIR / "sections.json", sections)
    qa_pairs = build_qa_pairs(sections)
    write_json(OUTPUT_DIR / "qa_pairs.json", qa_pairs)
    corpus = build_corpus(qa_pairs)
    chunks = chunk_corpus(corpus)
    vector_store = build_vector_store(chunks)
    run_sample_queries(vector_store)
    qa_chain = create_qa_chain(vector_store)
    launch_chat_interface(qa_chain)


if __name__ == "__main__":
    main()
