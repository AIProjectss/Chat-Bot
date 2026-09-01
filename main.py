import streamlit as st

from api import (
    SYSTEM_PROMPT,
    GroqChatError,
    get_chat_response,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>
    /* ---------- Global ---------- */

    .stApp {
        background: #0e1117;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* ---------- Header ---------- */

    .app-header {
        padding: 0.5rem 0 1.5rem 0;
        margin-bottom: 1rem;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 0.98rem;
        margin-top: 0;
    }

    .provider-badge {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #d1d5db;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* ---------- Empty State ---------- */

    .empty-state {
        text-align: center;
        padding: 6rem 1rem 4rem 1rem;
    }

    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .empty-title {
        font-size: 1.8rem;
        font-weight: 650;
        margin-bottom: 0.6rem;
    }

    .empty-description {
        color: #9ca3af;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0a0d12;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    .sidebar-brand {
        padding: 0.5rem 0 1.5rem 0;
    }

    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 700;
    }

    .sidebar-brand-subtitle {
        color: #8b949e;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .sidebar-section {
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .info-card {
        padding: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.025);
        margin-bottom: 0.6rem;
    }

    .info-label {
        color: #8b949e;
        font-size: 0.72rem;
        margin-bottom: 0.2rem;
    }

    .info-value {
        color: #e5e7eb;
        font-size: 0.86rem;
        font-weight: 600;
        word-break: break-word;
    }

    .about-text {
        color: #8b949e;
        font-size: 0.8rem;
        line-height: 1.6;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 0.7rem;
        min-height: 2.5rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatMessageContent"] code {
        border-radius: 0.35rem;
    }

    [data-testid="stChatMessageContent"] pre {
        border-radius: 0.7rem;
    }

    /* ---------- Chat Input ---------- */

    [data-testid="stChatInput"] {
        padding-bottom: 1rem;
    }

    /* ---------- Dividers ---------- */

    hr {
        border-color: rgba(255, 255, 255, 0.07);
    }

    /* ---------- Mobile ---------- */

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-title {
            font-size: 1.7rem;
        }

        .empty-state {
            padding-top: 4rem;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🤖 AI Assistant</div>
            <div class="sidebar-brand-subtitle">
                Fast conversational intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        st.rerun()

    st.markdown(
        '<div class="sidebar-section">Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Provider</div>
            <div class="info-value">Groq</div>
        </div>

        <div class="info-card">
            <div class="info-label">Inference</div>
            <div class="info-value">Groq LLM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Settings</div>',
        unsafe_allow_html=True,
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=st.session_state.temperature,
        step=0.1,
        help=(
            "Lower values produce more deterministic answers. "
            "Higher values produce more varied responses."
        ),
    )

    st.session_state.max_tokens = st.slider(
        "Maximum response tokens",
        min_value=256,
        max_value=8192,
        value=st.session_state.max_tokens,
        step=256,
        help="Maximum length allowed for a single assistant response.",
    )

    st.markdown(
        '<div class="sidebar-section">About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-text">
            A lightweight AI assistant powered by the Groq API
            and built with Python and Streamlit.
            <br><br>
            Designed with a modular architecture so additional
            capabilities can be added later.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">AI Assistant</div>
        <div class="app-subtitle">
            Fast, intelligent and conversational AI assistance.
        </div>
        <div class="provider-badge">
            Powered by Groq
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VISIBLE CHAT HISTORY
# ============================================================

visible_messages = [
    message
    for message in st.session_state.messages
    if message["role"] in {"user", "assistant"}
]


# ============================================================
# EMPTY STATE
# ============================================================

if not visible_messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">
                Welcome to AI Assistant
            </div>
            <div class="empty-description">
                Ask anything — from programming and science
                to writing, research and everyday questions.
                <br><br>
                How can I help you today?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

for message in visible_messages:

    role = message["role"]

    if role == "user":
        avatar = "👤"
    else:
        avatar = "✦"

    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message AI Assistant..."
)


if prompt:

    prompt = prompt.strip()

    if not prompt:
        st.warning("Please enter a message before sending.")
        st.stop()

    # --------------------------------------------------------
    # Add user message to session history
    # --------------------------------------------------------

    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(user_message)

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant", avatar="✦"):

        with st.spinner("Thinking..."):

            try:
                response = get_chat_response(
                    messages=st.session_state.messages,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )

            except GroqChatError as exc:
                st.error(str(exc))
                st.stop()

            except Exception:
                st.error(
                    "Something went wrong while generating the response. "
                    "Please try again."
                )
                st.stop()

        # ----------------------------------------------------
        # Display response
        # ----------------------------------------------------

        st.markdown(response)

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )