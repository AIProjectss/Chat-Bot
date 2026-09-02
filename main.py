import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

from api import (
    SYSTEM_PROMPT,
    GroqChatError,
    get_chat_response,
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


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
# SUPABASE CONNECTION
# ============================================================

def get_secret(name: str):
    """
    Get configuration from Streamlit Cloud secrets first,
    then fall back to local .env variables.
    """

    try:
        value = st.secrets.get(name)

        if value:
            return str(value)

    except Exception:
        pass

    return os.getenv(name)


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:

    st.error(
        "Supabase configuration is missing. "
        "Please configure SUPABASE_URL and SUPABASE_KEY."
    )

    st.stop()


@st.cache_resource
def get_supabase_client() -> Client:

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )


supabase = get_supabase_client()


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def create_conversation(title: str = "New Chat"):

    conversation_id = str(uuid.uuid4())

    result = (
        supabase
        .table("conversations")
        .insert(
            {
                "id": conversation_id,
                "title": title,
            }
        )
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Could not create a new conversation."
        )

    return result.data[0]


def get_conversations():

    result = (
        supabase
        .table("conversations")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )

    return result.data or []


def get_messages(conversation_id: str):

    result = (
        supabase
        .table("messages")
        .select("role, content, created_at")
        .eq(
            "conversation_id",
            conversation_id,
        )
        .order(
            "created_at",
            desc=False,
        )
        .execute()
    )

    return result.data or []


def save_message(
    conversation_id: str,
    role: str,
    content: str,
):

    result = (
        supabase
        .table("messages")
        .insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            }
        )
        .execute()
    )

    return result.data


def update_conversation_title(
    conversation_id: str,
    title: str,
):

    (
        supabase
        .table("conversations")
        .update(
            {
                "title": title,
            }
        )
        .eq(
            "id",
            conversation_id,
        )
        .execute()
    )


def update_conversation_timestamp(
    conversation_id: str,
):

    (
        supabase
        .table("conversations")
        .update(
            {
                "updated_at": "now()",
            }
        )
        .eq(
            "id",
            conversation_id,
        )
        .execute()
    )


def delete_conversation(
    conversation_id: str,
):

    (
        supabase
        .table("conversations")
        .delete()
        .eq(
            "id",
            conversation_id,
        )
        .execute()
    )


# ============================================================
# CONVERSATION TITLE
# ============================================================

def generate_title(prompt: str) -> str:

    title = " ".join(prompt.strip().split())

    if not title:
        return "New Chat"

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    return title


# ============================================================
# MATH NORMALIZATION
# ============================================================

def normalize_math_for_streamlit(text: str) -> str:

    if not text:
        return text

    text = re.sub(
        r"<br\s*/?>",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m:
            "\n\n$$\n"
            + m.group(1).strip()
            + "\n$$\n\n",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda m:
            "$"
            + m.group(1).strip()
            + "$",
        text,
        flags=re.DOTALL,
    )

    lines = text.splitlines()

    normalized_lines = []

    latex_indicators = (
        r"\mathbf",
        r"\operatorname",
        r"\text{",
        r"\boxed",
        r"\begin{",
        r"\end{",
        r"\frac",
        r"\sqrt",
        r"\sum",
        r"\prod",
        r"\sin",
        r"\cos",
        r"\log",
        r"\exp",
        r"\mathbb",
        r"\mathcal",
        r"\mathrm",
        r"\left",
        r"\right",
        r"\top",
        r"\cdot",
        r"\times",
    )

    for line in lines:

        stripped = line.strip()

        contains_latex = any(
            indicator in stripped
            for indicator in latex_indicators
        )

        already_math = (
            stripped.startswith("$$")
            or stripped.startswith("$")
            or stripped.startswith(r"\(")
            or stripped.startswith(r"\[")
        )

        if (
            contains_latex
            and stripped.startswith("[")
            and stripped.endswith("]")
            and not already_math
        ):

            equation = stripped[1:-1].strip()

            normalized_lines.append("$$")
            normalized_lines.append(equation)
            normalized_lines.append("$$")

            continue

        normalized_lines.append(line)

    return "\n".join(normalized_lines)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background: #FFFFFF;
    color: #1F1F1F;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ==========================================================
   HEADER
========================================================== */

.app-header {
    padding: 0.5rem 0 1.5rem 0;
    margin-bottom: 1rem;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #111827;
}

.app-subtitle {
    color: #6B7280;
    font-size: 0.98rem;
}

.provider-badge {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    color: #4B5563;
    font-size: 0.78rem;
    font-weight: 500;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    background: #F7F7F8;
    border-right: 1px solid #E5E7EB;
}

.sidebar-brand {
    padding: 0.5rem 0 1.5rem 0;
}

.sidebar-brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
}

.sidebar-brand-subtitle {
    color: #6B7280;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

.sidebar-section {
    margin-top: 1.5rem;
    margin-bottom: 0.7rem;
    color: #6B7280;
    text-transform: uppercase;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}


/* ==========================================================
   NEW CHAT BUTTON
========================================================== */

section[data-testid="stSidebar"] .stButton > button {
    background: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #F3F4F6 !important;
    border-color: #9CA3AF !important;
}


/* ==========================================================
   HISTORY BUTTONS
========================================================== */

.history-button button {
    text-align: left !important;
}


/* ==========================================================
   INFO CARD
========================================================== */

.info-card {
    padding: 0.8rem;
    border: 1px solid #E5E7EB;
    border-radius: 0.75rem;
    background: #FFFFFF;
    margin-bottom: 0.6rem;
}

.info-label {
    color: #6B7280;
    font-size: 0.72rem;
}

.info-value {
    color: #1F2937;
    font-size: 0.86rem;
    font-weight: 600;
}


/* ==========================================================
   CHAT
========================================================== */

[data-testid="stChatMessage"] {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="stChatMessageContent"] {
    line-height: 1.7;
}

[data-testid="stChatMessageContent"] p {
    margin-bottom: 0.7rem;
    color: #1F1F1F;
}

[data-testid="stChatMessageContent"] code {
    border-radius: 0.35rem;
}

[data-testid="stChatMessageContent"] pre {
    border-radius: 0.7rem;
}


/* ==========================================================
   CHAT INPUT
========================================================== */

[data-testid="stChatInput"] {
    padding-bottom: 1rem;
}

[data-testid="stChatInput"] textarea {
    color: #1F1F1F !important;
    background: #FFFFFF !important;
}


/* ==========================================================
   EMPTY STATE
========================================================== */

.empty-state {
    text-align: center;
    padding: 6rem 1rem 4rem 1rem;
}

.empty-icon {
    font-size: 3rem;
}

.empty-title {
    font-size: 1.8rem;
    font-weight: 650;
    color: #111827;
}

.empty-description {
    color: #6B7280;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 auto;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {
    border-radius: 0.7rem;
    min-height: 2.5rem;
    font-weight: 600;
    transition: all 0.15s ease;
}


/* ==========================================================
   DIVIDERS
========================================================== */

hr {
    border-color: #E5E7EB;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


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


if "current_conversation_id" not in st.session_state:

    st.session_state.current_conversation_id = None


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

            <div class="sidebar-brand-title">
                🤖 AI Assistant
            </div>

            <div class="sidebar-brand-subtitle">
                Fast conversational intelligence
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="secondary",
    ):

        st.session_state.current_conversation_id = None

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        st.rerun()


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Chat History</div>',
        unsafe_allow_html=True,
    )


    try:

        conversations = get_conversations()

    except Exception as exc:

        st.error(
            "Could not load chat history."
        )

        conversations = []


    if conversations:

        for conversation in conversations:

            conversation_id = conversation["id"]

            title = conversation.get(
                "title",
                "New Chat",
            )

            # Short display title
            display_title = title

            if len(display_title) > 28:
                display_title = (
                    display_title[:28].rstrip()
                    + "..."
                )


            col1, col2 = st.columns(
                [5, 1],
                gap="small",
            )


            with col1:

                if st.button(
                    f"💬 {display_title}",
                    key=f"conversation_{conversation_id}",
                    use_container_width=True,
                ):

                    try:

                        loaded_messages = get_messages(
                            conversation_id
                        )

                        st.session_state.current_conversation_id = (
                            conversation_id
                        )

                        st.session_state.messages = [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            }
                        ]

                        for message in loaded_messages:

                            st.session_state.messages.append(
                                {
                                    "role": message["role"],
                                    "content": message["content"],
                                }
                            )

                        st.rerun()

                    except Exception:

                        st.error(
                            "Could not load this conversation."
                        )


            with col2:

                if st.button(
                    "⋮",
                    key=f"delete_{conversation_id}",
                    help="Delete conversation",
                ):

                    try:

                        delete_conversation(
                            conversation_id
                        )

                        if (
                            st.session_state.current_conversation_id
                            == conversation_id
                        ):

                            st.session_state.current_conversation_id = None

                            st.session_state.messages = [
                                {
                                    "role": "system",
                                    "content": SYSTEM_PROMPT,
                                }
                            ]

                        st.rerun()

                    except Exception:

                        st.error(
                            "Could not delete conversation."
                        )

    else:

        st.caption(
            "No previous conversations yet."
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Provider
            </div>

            <div class="info-value">
                Groq
            </div>

        </div>

        <div class="info-card">

            <div class="info-label">
                Model
            </div>

            <div class="info-value">
                GPT-OSS 120B
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

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
    )


    st.session_state.max_tokens = st.slider(
        "Maximum response tokens",
        min_value=256,
        max_value=8192,
        value=st.session_state.max_tokens,
        step=256,
    )


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="about-text">

            A lightweight AI assistant powered by
            Groq and built with Python + Streamlit.

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

        <div class="app-title">
            AI Assistant
        </div>

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
# VISIBLE MESSAGES
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

            <div class="empty-icon">
                ✦
            </div>

            <div class="empty-title">
                Welcome to AI Assistant
            </div>

            <div class="empty-description">

                Ask anything — from programming and
                science to writing, research and
                everyday questions.

                <br><br>

                How can I help you today?

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RENDER CHAT
# ============================================================

for message in visible_messages:

    role = message["role"]

    with st.chat_message(role):

        content = message["content"]

        if role == "assistant":

            content = normalize_math_for_streamlit(
                content
            )

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message AI Assistant..."
)


if prompt:

    prompt = prompt.strip()

    if not prompt:

        st.warning(
            "Please enter a message before sending."
        )

        st.stop()


    # ========================================================
    # CREATE CONVERSATION IF NECESSARY
    # ========================================================

    if not st.session_state.current_conversation_id:

        try:

            conversation = create_conversation(
                title=generate_title(prompt)
            )

            st.session_state.current_conversation_id = (
                conversation["id"]
            )

        except Exception as exc:

            st.error(
                f"Could not create conversation: {exc}"
            )

            st.stop()


    conversation_id = (
        st.session_state.current_conversation_id
    )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    try:

        save_message(
            conversation_id=conversation_id,
            role="user",
            content=prompt,
        )

    except Exception as exc:

        st.error(
            f"Could not save your message: {exc}"
        )

        st.stop()


    # ========================================================
    # ADD USER MESSAGE TO SESSION
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    # ========================================================
    # GENERATE ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = get_chat_response(
                    messages=st.session_state.messages,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )

            except GroqChatError as exc:

                st.error(
                    str(exc)
                )

                st.stop()

            except Exception as exc:

                st.error(
                    f"Something went wrong: {exc}"
                )

                st.stop()


        response = normalize_math_for_streamlit(
            response
        )

        st.markdown(
            response
        )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    try:

        save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
        )

        update_conversation_timestamp(
            conversation_id=conversation_id,
        )

    except Exception as exc:

        st.warning(
            f"Response generated, but history could not "
            f"be saved: {exc}"
        )


    # ========================================================
    # ADD ASSISTANT RESPONSE TO SESSION
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
