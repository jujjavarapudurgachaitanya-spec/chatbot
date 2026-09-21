"""Streamlit chat UI:  streamlit run app.py"""
import os
import subprocess

# Check if the database file exists; if not, create it
if not os.path.exists("data/chatbot.db"):
    subprocess.run(["python", "data/generate_sample_data.py"])

import sqlite3

import streamlit as st

import chatbot
import guardrails as g

st.set_page_config(page_title="Text-to-SQL", page_icon=":bar_chart:")
st.title("Text-to-SQL chatbot")
st.caption(f"Read-only over {chatbot.DB.name} - {len(chatbot.tables())} tables, "
           f"SELECT only, max {g.ROW_LIMIT} rows")

history = st.session_state.setdefault("history", [])
for turn in history:
    st.chat_message("user").write(turn["q"])
    with st.chat_message("assistant"):
        st.code(turn["sql"], language="sql")
        if turn["df"] is not None:
            st.dataframe(turn["df"])
        else:
            st.error(turn["sql"])

if question := st.chat_input("e.g. top 10 vendors by total PO value"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Writing SQL..."):
        try:
            sql, df = chatbot.ask(question)
            st.code(sql, language="sql")
            st.dataframe(df)
            history.append({"q": question, "sql": sql, "df": df})
        except (g.Blocked, sqlite3.Error) as e:
            st.error(str(e))
            history.append({"q": question, "sql": str(e), "df": None})
