"""Streamlit travel assistant gated by Firebase Authentication."""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from firebase_client import AuthError, sign_in, sign_up
from rag import RagError, UNKNOWN, build_index, generate_answer, retrieve

# Reload local configuration on each Streamlit rerun after the student edits .env.
load_dotenv(override=True)
st.set_page_config(page_title='Lumen Travel Assistant', page_icon='🧭', layout='centered')

OLLAMA_URL = os.getenv('OLLAMA_URL', '').strip().rstrip('/')
GUIDE_PATH = str(Path(__file__).with_name('knowledge_base.json'))

try:
    TOP_K = int(os.getenv('TOP_K') or '3')
    MIN_SIMILARITY = float(os.getenv('MIN_SIMILARITY') or '0.0')
    if not 1 <= TOP_K <= 10:
        raise ValueError('TOP_K must be between 1 and 10.')
    if not -1.0 <= MIN_SIMILARITY <= 1.0:
        raise ValueError('MIN_SIMILARITY must be between -1.0 and 1.0.')
except ValueError as exc:
    st.error(f'Invalid .env configuration: {exc}')
    st.stop()


@st.cache_resource(show_spinner='Embedding the guide (once per server/guide version)...')
def get_index(ollama_url: str, guide_path: str, guide_mtime: float):
    # guide_mtime automatically invalidates the cache after editing the JSON file.
    return build_index(ollama_url, guide_path)


def set_user(firebase_response: dict):
    st.session_state['user'] = {
        'email': firebase_response['email'],
        'id_token': firebase_response['idToken'],
        'expires_at': time.time() + max(1, int(firebase_response.get('expiresIn', 3600))) - 30,
    }
    st.session_state['messages'] = []


def sign_out():
    st.session_state.clear()  # Includes chat history. No Firestore/database.
    st.rerun()


def logged_in() -> bool:
    user = st.session_state.get('user')
    return bool(user and user.get('id_token') and time.time() < user.get('expires_at', 0))


st.title('🧭 Lumen Travel')
st.caption('Ho Chi Minh City · Guide-grounded RAG chatbot')

if not logged_in():
    # Explicitly drop any expired user state and associated chat history.
    if st.session_state.get('user'):
        st.session_state.clear()
        st.warning('Your login expired. Please sign in again.')

    st.info('Sign in to access the travel assistant.')
    tab_in, tab_up = st.tabs(['Sign in', 'Create account'])
    with tab_in:
        with st.form('sign_in_form'):
            email = st.text_input('Email', key='in_email')
            password = st.text_input('Password', type='password', key='in_pass')
            submit_in = st.form_submit_button('Sign in', type='primary')
        if submit_in:
            try:
                set_user(sign_in(email, password))
                st.rerun()
            except AuthError as exc:
                st.error(str(exc))
    with tab_up:
        with st.form('sign_up_form'):
            new_email = st.text_input('Email', key='up_email')
            new_password = st.text_input('Password (at least 6 characters)', type='password', key='up_pass')
            confirm = st.text_input('Confirm password', type='password', key='up_confirm')
            submit_up = st.form_submit_button('Create account')
        if submit_up:
            if new_password != confirm:
                st.error('Passwords do not match.')
            elif len(new_password) < 6:
                st.error('Password must be at least 6 characters.')
            else:
                try:
                    set_user(sign_up(new_email, new_password))
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
    st.stop()  # Non-authenticated users cannot see or access the chat UI.

with st.sidebar:
    st.success(f"Signed in as: {st.session_state['user']['email']}")
    if st.button('Sign out', use_container_width=True):
        sign_out()
    st.caption(f'Models: embeddinggemma + qwen3:4b · top-k = {TOP_K}')

if not OLLAMA_URL:
    st.error('OLLAMA_URL is missing. Add the Pinggy URL to your .env file.')
    st.stop()

try:
    entries, vectors = get_index(OLLAMA_URL, GUIDE_PATH, Path(GUIDE_PATH).stat().st_mtime)
except (RagError, OSError, ValueError) as exc:
    st.error(f'Cannot prepare the guide: {exc}')
    st.stop()

st.markdown(f'**Travel guide:** {len(entries)} entries. Ask about landmarks, markets, food, or transport.')


def render_sources(sources: list[dict]):
    with st.expander(f'Retrieved entries ({len(sources)}) · cosine similarity'):
        for entry in sources:
            st.markdown(f"**[{entry['id']}] {entry['title']}** — score: `{entry['score']:.3f}`")
            st.write(entry['text'])
            st.caption(f"Source: {entry['source']}")


for message in st.session_state.setdefault('messages', []):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            render_sources(message['sources'])

question = st.chat_input('Ask a question about Ho Chi Minh City...')
if question is not None:
    if not question.strip():
        st.warning('Please enter a non-empty question.')
        st.stop()
    st.session_state['messages'].append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        try:
            with st.spinner('Retrieving sources and generating an answer...'):
                sources = retrieve(OLLAMA_URL, question, entries, vectors, k=TOP_K)
                answer = generate_answer(OLLAMA_URL, question, sources, min_similarity=MIN_SIMILARITY)
            st.markdown(answer)
            render_sources(sources)
            st.session_state['messages'].append({
                'role': 'assistant', 'content': answer, 'sources': sources,
            })
        except RagError as exc:
            st.error(str(exc))
            st.session_state['messages'].append({
                'role': 'assistant', 'content': f'⚠️ {exc}', 'sources': [],
            })
