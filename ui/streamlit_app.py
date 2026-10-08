import uuid, requests, streamlit as st
import os

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
    r = requests.post(API, json={"session_id": st.session_state.sid, "question": q}).json()
    with st.chat_message("assistant"):
        st.write(r["answer"])
        if r["sources"]:
            with st.expander("Sources"):
                st.caption(f"Searched for: {r['rewritten']}")
                for s in r["sources"]:
                    st.write(f"{s['file']} (similarity {s['score']})")
    st.session_state.msgs.append({"role": "assistant", "text": r["answer"]})