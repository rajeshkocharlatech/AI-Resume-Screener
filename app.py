import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorstore, run_self_query
import os

st.set_page_config(page_title="AI Resume Screener")
st.title("AI Resume Screener")
st.markdown("Upload a resume and analyze it using AI. Then run smart searches over previous resumes.")

job_desc = st.text_area("Paste Job Description")
uploaded_file = st.file_uploader("📎 Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze & Store") and uploaded_file and job_desc:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing & Storing Resume..."):
        docs = load_resume(uploaded_file.name)
        report = analyze_resume(docs, job_desc)  # uses Gemini inside resume_processor.py
        store_to_vectorstore(docs)
        st.success("✅ Analysis complete and stored!")

        st.subheader("📄 AI Resume Summary")
        st.write(report)
        st.download_button("📥 Download Report", report, file_name="resume_analysis.txt")

st.divider()
st.subheader("🔎 Ask Anything About Stored Resumes")

query = st.text_input("Type your smart query here (e.g., 'Python developer with AWS')")

if st.button("Search Resumes") and query:
    with st.spinner("Searching..."):
        results = run_self_query(query)
        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.write(res.page_content.strip())
        else:
            st.warning("No matches found.")