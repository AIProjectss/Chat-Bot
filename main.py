import streamlit as st

from api import (
    SYSTEM_PROMPT,
    GroqChatError,
    get_chat_response,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.markdown("""
<style>

/* ================================
   GLOBAL APP
================================ */

.stApp {
    background-color: #FFFFFF;
    color: #1F1F1F;
}

/* Main content area */
.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF !important;
    color: #1F1F1F !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

/* Hover */
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    border-color: #9CA3AF !important;
}

/* Click / focus */
section[data-testid="stSidebar"] .stButton > button:focus {
    background-color: #FFFFFF !important;
    color: #1F1F1F !important;
    border-color: #9CA3AF !important;
    box-shadow: none !important;
}


/* ================================
   CHAT MESSAGES
================================ */

[data-testid="stChatMessage"] {
    background-color: transparent;
    border: none;
    padding: 1rem 0;
}


/* User message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background-color: #F4F4F5;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}


/* Assistant message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background-color: #FFFFFF;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}


/* ================================
   CHAT TEXT
================================ */

[data-testid="stChatMessage"] p {
    color: #1F1F1F;
    line-height: 1.65;
    font-size: 15px;
}


/* Headings inside responses */

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #111827;
}


/* ================================
   CHAT INPUT
================================ */

[data-testid="stChatInput"] {
    background-color: #FFFFFF;
}

[data-testid="stChatInput"] > div {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #1F1F1F !important;
    background-color: #FFFFFF !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9CA3AF !important;
}


/* ================================
   BUTTONS
================================ */

.stButton > button {
    background-color: #FFFFFF;
    color: #1F1F1F;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #F3F4F6;
    border-color: #9CA3AF;
}


/* ================================
   PRIMARY BUTTON
================================ */

.stButton > button[kind="primary"] {
    background-color: #1F2937;
    color: #FFFFFF;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    background-color: #111827;
}


/* ================================
   SELECTBOX
================================ */

div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #D1D5DB;
    color: #1F1F1F;
}


/* ================================
   EXPANDERS
================================ */

[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}


/* ================================
   MARKDOWN / GENERAL TEXT
================================ */

.stMarkdown {
    color: #1F1F1F;
}


/* ================================
   CODE BLOCKS
================================ */

[data-testid="stCodeBlock"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}


/* ================================
   DIVIDERS
================================ */

hr {
    border-color: #E5E7EB;
}


/* ================================
   SCROLLBAR
================================ */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #FFFFFF;
}

::-webkit-scrollbar-thumb {
    background: #D1D5DB;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF;
}

</style>
""", unsafe_allow_html=True)


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

    if role not in ["user", "assistant"]:
        continue

    with st.chat_message(role):
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

    with st.chat_message("assistant"):

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
