import os
from itertools import zip_longest
import textwrap

import streamlit as st
import emoji
from dotenv import load_dotenv
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
#  BASIC SETUP
# ========================

load_dotenv()  # not strictly needed now, but harmless

PDF_DIR = os.path.join("data", "books")  # folder with your PDFs


st.set_page_config(page_title="Offline Financial Advisor", page_icon="💸")

st.title("💸 Offline Financial Advisor (no API keys)")
st.write(
    "This app is **offline** – no OpenAI, no API keys.\n\n"
    "It reads your finance PDFs, finds relevant passages, and wraps them in basic advice.\n"
    "**Educational only – not professional financial advice.**"
)

# ========================
#  SIDEBAR INPUT
# ========================

def get_user_question():
    with st.sidebar:
        st.header("Ask a financial question")
        q = st.text_input(
            "Your question:",
            value="How should I start investing as a beginner?",
            key="user_input",
        )
        send = st.button("Send")
    if send and q.strip():
        return q.strip()
    return None


user_question = get_user_question()

# ========================
#  SESSION STATE
# ========================

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

# ========================
#  PDF LOADING & CHUNKING
# ========================

def read_pdfs(pdf_dir: str):
    """Read all PDFs in a folder and return a single big text string."""
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    texts = []
    for filename in files:
        path = os.path.join(pdf_dir, filename)
        try:
            reader = PdfReader(path)
            txt = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    txt += page_text + "\n"
            texts.append(txt)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """
    Split long text into overlapping chunks for retrieval.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@st.cache_resource(show_spinner=True)
def build_knowledge_base(pdf_dir: str):
    """
    1. Read PDFs
    2. Chunk text
    3. Create embeddings for each chunk
    """
    raw_text = read_pdfs(pdf_dir)
    chunks = chunk_text(raw_text, chunk_size=800, overlap=200)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    return model, chunks, embeddings


# ========================
#  RETRIEVAL + ANSWER
# ========================

def retrieve_relevant_chunks(model, chunks, embeddings, query: str, k: int = 3):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[::-1][:k]
    results = [(chunks[i], float(sims[i])) for i in top_idx]
    return results


def classify_topic(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["debt", "loan", "credit card", "borrow"]):
        return "debt"
    if any(w in q for w in ["budget", "spend", "expense", "saving", "save"]):
        return "budget"
    if any(w in q for w in ["invest", "stock", "etf", "mutual fund"]):
        return "investing"
    if any(w in q for w in ["crypto", "bitcoin", "ethereum", "altcoin"]):
        return "crypto"
    return "general"


def advisor_intro(topic: str) -> str:
    if topic == "debt":
        return (
            "Focus on high-interest debt first, avoid new borrowing, "
            "and build at least a small emergency buffer so you don’t fall back into debt."
        )
    if topic == "budget":
        return (
            "A simple starting point is the 50/30/20 rule: "
            "about 50% needs, 30% wants, 20% savings and debt payments – "
            "adjusted to your income and cost of living."
        )
    if topic == "investing":
        return (
            "Long-term, diversified investing (for example broad index funds) "
            "is usually safer than chasing hot tips or short-term trades."
        )
    if topic == "crypto":
        return (
            "Treat crypto as highly speculative: only money you can afford to lose, "
            "never your emergency fund, and never 100% of your investments."
        )
    return (
        "Good financial decisions usually start with an emergency fund, "
        "controlled spending, and clear long-term goals."
    )


def build_answer(question: str, retrieved):
    """
    Wrap retrieved book passages in simple advisor-style text.
    """
    topic = classify_topic(question)
    intro = advisor_intro(topic)

    snippets = []
    for chunk, score in retrieved:
        cleaned = " ".join(chunk.split())
        # shorten each snippet
        shortened = textwrap.shorten(cleaned, width=450, placeholder=" ...")
        snippets.append(f"- {shortened}")

    if not snippets:
        body = "I couldn't find anything clearly relevant in the book for this question."
    else:
        body = (
            "Here are some ideas related to your question, based on the book content:\n\n"
            + "\n".join(snippets)
        )

    answer = (
        f"**Your question:** {question}\n\n"
        f"**Advisor perspective:** {intro}\n\n"
        f"{body}\n\n"
        "_Remember: this is general educational guidance, not personalised professional advice._"
    )
    return answer


# ========================
#  MAIN FLOW
# ========================

if user_question:
    try:
        with st.spinner("Building knowledge base from your PDFs (first run may take a bit)..."):
            model, chunks, embeddings = build_knowledge_base(PDF_DIR)

        retrieved = retrieve_relevant_chunks(model, chunks, embeddings, user_question, k=3)
        response = build_answer(user_question, retrieved)

        st.session_state["past"].append(user_question)
        st.session_state["generated"].append(response)

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ========================
#  DISPLAY CONVERSATION
# ========================

if st.session_state["generated"]:
    with st.expander("💬 Conversation", expanded=True):
        for i in range(len(st.session_state["generated"])):
            user_msg = st.session_state["past"][i]
            bot_msg = st.session_state["generated"][i]

            st.markdown(
                emoji.emojize(f":speech_balloon: **You:** {user_msg}")
            )
            st.markdown(
                emoji.emojize(f":robot: **Advisor:** {bot_msg}")
            )
else:
    st.info("Ask a question in the sidebar to start the conversation.")
