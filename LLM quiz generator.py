import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyPDF2 import PdfReader
from docx import Document
import torch

from langchain.schema import Document as LC_Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔧 Load your LLaMA model
@st.cache_resource
def load_amal_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

# 📄 Read files
def amal_read_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return None

# ✂ Split content into chunks
def split_text_with_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = text_splitter.create_documents([text])
    return docs

# 🧠 Generate quiz for each chunk
def amal_generate_mcqs_llama(generator, chunks, num_questions=5):
    all_quizzes = []

    for i, chunk in enumerate(chunks):
        prompt = (
            f"You are a professional quiz generator. From the following content, create {num_questions} multiple-choice questions "
            f"with 4 options and the correct answer marked clearly.\n\n"
            f"Content:\n{chunk.page_content}"
        )
        response = generator(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)[0]["generated_text"]
        quiz_part = response.split("Content:")[-1].strip()
        all_quizzes.append(f"📚 Chunk {i+1} Quiz:\n{quiz_part}")

    return "\n\n".join(all_quizzes)

# 🌐 Streamlit UI
st.set_page_config(page_title="Amal's LLaMA + LangChain Quiz Generator", layout="wide")
st.title("🧠 Amal's LLaMA-Powered Quiz Generator with LangChain")

uploaded_file = st.file_uploader("📤 Upload your study material (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
num_questions = st.slider("📌 Questions per Chunk", 1, 5, 3)

if uploaded_file:
    with st.spinner("📖 Reading file..."):
        full_text = amal_read_file(uploaded_file)

    if full_text:
        with st.spinner("🔍 Splitting text with LangChain..."):
            chunks = split_text_with_langchain(full_text)

        st.success(f"✅ Text split into {len(chunks)} chunks")

        generator = load_amal_model()

        if st.button("🚀 Generate MCQs"):
            with st.spinner("🧠 Generating quiz using LLaMA..."):
                final_quiz = amal_generate_mcqs_llama(generator, chunks, num_questions)
                st.markdown("### ✅ Your Generated MCQs")
                st.text_area("📝 Quiz Output", final_quiz, height=500)
    else:
        st.error("⚠ Unable to read file.")