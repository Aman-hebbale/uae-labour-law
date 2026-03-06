import streamlit as st
import requests
import os

# BACKEND_URL = os.getenv("BACKEND_URL")
# BACKEND_URL = os.getenv("BACKEND_URL")

st.title("GCP RAG Assistant")

query = st.text_input("Ask me anything about your data:")
# res = requests.get(f"http://136.116.16.132:8000/ask", params={"query": "what is wage"})
# print(res.json())
if st.button("Send"):
    res = requests.get(f"http://localhost:8000/ask", params={"query": query})
    st.write(res.json().get("answer"))