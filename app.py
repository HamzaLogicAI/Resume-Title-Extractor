import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # <-- Must be set before importing transformers

import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import streamlit.components.v1 as components

# Set page configuration with a modern layout
st.set_page_config(page_title="Resume Job Title Extractor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with Tailwind for professional styling
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    
    .main-container {
        @apply max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg;
    }
    .title {
        @apply text-4xl font-bold text-center text-gray-800 mb-4;
    }
    .subtitle {
        @apply text-lg text-gray-600 text-center mb-6;
    }
    .section-header {
        @apply text-2xl font-semibold text-gray-700 mt-8 mb-4;
    }
    .upload-box {
        @apply border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition;
    }
    .success-box {
        @apply bg-green-100 border-l-4 border-green-500 p-4 rounded-lg;
    }
    .error-box {
        @apply bg-red-100 border-l-4 border-red-500 p-4 rounded-lg;
    }
    .download-btn {
        @apply bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition;
    }
    </style>
""", unsafe_allow_html=True)

# Lottie animation for visual flair
def load_lottie_animation():
    lottie_html = """
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_3pgxyh.json" background="transparent" speed="1" style="width: 300px; height: 300px; margin: auto;" loop autoplay></lottie-player>
    """
    components.html(lottie_html, height=300)

# Main container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header with title and subtitle
    st.markdown('<h1 class="title">📄 Resume Extractor + 🧠 Job Title Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a resume PDF to extract the job title using advanced NLP.</p>', unsafe_allow_html=True)
    
    # Lottie animation
    
    # File uploader with styled box
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("📄 Processing resume..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = "".join([page.get_text("text") for page in doc])

        # Two-column layout for extracted text and download button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<h2 class="section-header">📜 Extracted Resume Text</h2>', unsafe_allow_html=True)
            st.text_area("Resume Content", resume_text[:3000], height=300, key="resume_text")
        
        with col2:
            st.markdown('<div class="mt-10"></div>', unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Extracted Text",
                data=resume_text,
                file_name="resume_text.txt",
                mime="text/plain",
                help="Download the extracted resume text as a .txt file",
                key="download_btn",
                use_container_width=True,
            )

        if resume_text.strip():
            st.markdown('<h2 class="section-header">💼 Predicted Job Title</h2>', unsafe_allow_html=True)
            try:
                qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
                result = qa(question="What is the job title of this person?", context=resume_text)
                st.markdown(f'<div class="success-box"><p class="text-lg font-semibold text-green-700">{result["answer"]}</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-box"><p class="text-lg font-semibold text-red-700">❌ Error: {e}</p></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
        This app uses **PyMuPDF** to extract text from uploaded PDF resumes and **Transformers** to predict the job title using NLP. 
        Upload a PDF resume to see the magic happen! 🚀
    """)
    st.markdown("**Built with ❤️ by Streamlit**")
