import streamlit as st

st.title("GCP Agentic Chatbot")
st.write("This is my interview-ready frontend!")

if prompt := st.chat_input("Say something"):
    st.chat_message("user").markdown(prompt)
    st.chat_message("assistant").markdown(f"Echo: {prompt}")