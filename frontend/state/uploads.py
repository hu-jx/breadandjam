import streamlit as st
from models.upload import Upload

_UPLOADS_KEY = "uploads"
_IDX_KEY = "current_idx"

def init():
    st.session_state.setdefault(_UPLOADS_KEY, [])
    st.session_state.setdefault(_IDX_KEY, 0)

def all_uploads():
    return st.session_state[_UPLOADS_KEY]

def add(upload: Upload):
    st.session_state[_UPLOADS_KEY].append(upload)

def exists(name: str):
    return any(u.name == name for u in all_uploads())

def current():
    uploads = all_uploads()
    if (not uploads):
        return None
    return uploads[st.session_state[_IDX_KEY] % len(uploads)]

def next_image():
    uploads = all_uploads()
    if (uploads):
        st.session_state[_IDX_KEY] = (st.session_state[_IDX_KEY] + 1) % len(uploads)

def prev_image():
    uploads = all_uploads()
    if (uploads):
        st.session_state[_IDX_KEY] = (st.session_state[_IDX_KEY] - 1) % len(uploads)

def recent(limit: int = 10):
    return list(reversed(all_uploads()[-limit:]))

def stats():
    uploads = all_uploads()
    total = len(uploads)
    flagged = sum(1 for u in uploads if u.verdict == "AI")
    return {"total": total, "flagged": flagged}

def current_position():
    uploads = all_uploads()
    if not uploads:
        return 0, 0
    
    index = st.session_state[_IDX_KEY] % len(uploads)
    return index + 1, len(uploads)