import streamlit as st
from response import  loadModel, summarizeTextWithoutChunking,approach1, approach2, approach3, approach4, recursive_bart_summarization

with st.spinner("loading model..."):
    loadModel()
    st.success("model loaded!")

inputText = st.text_area("Text to be summarized")

if st.button("Summarize") and inputText!= None:
    with st.spinner("summarizing text..."):
        output = approach1(inputText)
        st.success(output)
