import os
import uuid

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
st.title("HR Policy Assistant")

if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.write(m["text"])

if q := st.chat_input("Ask about leave, WFH, expenses..."):
    st.session_state.msgs.append({"role": "user", "text": q})
    with st.chat_message("user"):
        st.write(q)

    try:
        resp = requests.post(
            API,
            json={"session_id": st.session_state.sid, "question": q},
            timeout=60,
        )
        resp.raise_for_status()
        r = resp.json()
    except requests.RequestException:
        st.error("The assistant is unavailable right now. Please try again.")
        st.stop()

    with st.chat_message("assistant"):
        st.write(r["answer"])
        if r["sources"]:
            with st.expander("Retrieved context"):
                st.caption(f"Searched for: {r['rewritten']}")
                best = {}
                for s in r["sources"]:
                    best[s["file"]] = max(best.get(s["file"], 0), s["score"])
                for f, score in best.items():
                    st.write(f"{f} (best similarity {score})")
    st.session_state.msgs.append({"role": "assistant", "text": r["answer"]})