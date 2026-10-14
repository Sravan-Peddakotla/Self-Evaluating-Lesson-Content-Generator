"""Streamlit interface for the self-evaluating lesson generator."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from lesson_engine import LessonMemory, run_pipeline

load_dotenv()


st.set_page_config(
    page_title="Lesson Foundry | AI Teacher",
    page_icon="LF",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#202725; --paper:#f5f3ed; --muted:#727a74; --line:#d9d9cf; --coral:#c94f35; --green:#476b50; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#202725; }
[data-testid="stSidebar"] * { color:#f5f3ed; }
[data-testid="stSidebar"] input { color:#202725 !important; }
h1, h2, h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:0 !important; color:var(--ink); }
h1 { font-size:2.35rem !important; line-height:1.1 !important; }
.eyebrow { color:var(--coral); font-family:'DM Mono',monospace; font-size:.74rem; text-transform:uppercase; }
.lede { max-width:760px; color:#58615b; font-size:1.04rem; line-height:1.6; }
.status-pass { color:#315c3d; background:#e2ebdf; padding:.4rem .7rem; display:inline-block; font-family:'DM Mono',monospace; font-size:.78rem; }
.status-fail { color:#923b2a; background:#f4e2dc; padding:.4rem .7rem; display:inline-block; font-family:'DM Mono',monospace; font-size:.78rem; }
.stButton > button[kind="primary"] { background:var(--coral); border-color:var(--coral); color:#fff; }
.stButton > button[kind="primary"]:hover { background:#fff; border-color:#202725; color:#202725; }
.stButton > button { border-radius:8px; }
div[data-testid="stTextArea"] textarea { background:#fffefa; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_memory() -> LessonMemory:
    return LessonMemory()


memory = get_memory()


def render_run(run: dict) -> None:
    st.markdown(f"<span class='eyebrow'>Run · {run['provider']} · {run['model']}</span>", unsafe_allow_html=True)
    if run["passed"]:
        st.markdown("<span class='status-pass'>QUALITY GATE PASSED</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-fail'>DID NOT PASS · REVIEW BEFORE SHARING</span>", unsafe_allow_html=True)
    st.markdown(f"### {run['topic']}")

    latest = run["attempts"][-1]
    passed_count = sum(check["pass"] for check in latest["evaluation"]["checks"])
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Drafts", len(run["attempts"]))
    col_b.metric("Final checkpoints", f"{passed_count}/{len(latest['evaluation']['checks'])}")
    col_c.metric("Retries used", len(run["attempts"]) - 1)

    lesson_tab, rubric_tab, log_tab = st.tabs(["Lesson", "Rubric", "Rejection log"])
    with lesson_tab:
        st.markdown(run["lesson"])
        st.download_button(
            "Download lesson as Markdown",
            data=run["lesson"],
            file_name="beginner-lesson.md",
            mime="text/markdown",
            key=f"download-{run['id']}",
        )
    with rubric_tab:
        for check in latest["evaluation"]["checks"]:
            icon = "PASS" if check["pass"] else "FAIL"
            suffix = f" — {check['reason']}" if not check["pass"] else ""
            st.markdown(f"**{icon} · {check['label']}**{suffix}")
    with log_tab:
        if len(run["attempts"]) == 1 and run["passed"]:
            st.info("The first draft passed every hard checkpoint. No rejection or retry was needed.")
        for attempt in run["attempts"]:
            evaluation = attempt["evaluation"]
            with st.expander(f"Draft {attempt['attempt']} · {'passed' if evaluation['passed'] else 'rejected'}", expanded=not evaluation["passed"]):
                failures = [check for check in evaluation["checks"] if not check["pass"]]
                if failures:
                    for check in failures:
                        st.markdown(f"- **{check['label']}:** {check['reason']}")
                    if attempt["attempt"] < len(run["attempts"]):
                        st.caption("These rejection reasons were sent to the generator as required fixes for the next draft.")
                else:
                    st.write("All checkpoints passed; this draft cleared the quality gate.")
                if evaluation["retry_guidance"]:
                    st.markdown("**Evaluator's suggested changes**")
                    for suggestion in evaluation["retry_guidance"]:
                        st.markdown(f"- {suggestion}")


with st.sidebar:
    st.markdown("## Lesson Foundry")
    st.caption("Generate · judge · improve")
    providers = ["Groq", "Google Gemini", "OpenAI"]
    provider = st.selectbox("Model provider", providers, index=0, key="provider_v2")
    env_key_name = {
        "Groq": "GROQ_API_KEY",
        "Google Gemini": "GEMINI_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
    }[provider]
    env_key = os.getenv(env_key_name, "")
    try:
        env_key = env_key or st.secrets.get(env_key_name, "")
    except (FileNotFoundError, AttributeError):
        pass
    api_key = env_key
    st.info("Using free-tier API key. Results may not be appropriate.")
    default_model = {
        "Groq": "openai/gpt-oss-120b",
        "Google Gemini": "gemini-3.8-flash",
        "OpenAI": "gpt-4o-mini",
    }[provider]
    model = st.text_input("Model", value=default_model, key=f"model_name_v2_{provider.replace(' ', '_')}")
    max_retries = st.select_slider("Maximum retries", options=[0, 1, 2], value=2)
    st.divider()
    st.caption("The selected provider and model handle both generation and evaluation. Each draft receives six binary checks; at most two retries are allowed.")
    with st.expander("Persistent memory"):
        guidance = memory.adaptive_guidance()
        st.write(guidance or "No recurring failures yet. Failure patterns will appear after they repeat across runs.")
        st.caption(f"Saved runs: {len(memory.recent_runs(1000))}")

st.markdown("<span class='eyebrow'>AI Teacher · Content Systems</span>", unsafe_allow_html=True)
st.title("A lesson that earns its way out.")
st.markdown("<p class='lede'>Create a clear first lesson for a learner starting from zero. An evaluator checks every hard requirement, sends failures back for revision, and keeps a record of what changed.</p>", unsafe_allow_html=True)

topic_col, button_col = st.columns([5, 1.25], vertical_alignment="bottom")
with topic_col:
    topic = st.text_input("Lesson topic", value="Introduction to RAG", placeholder="e.g. How neural networks learn")
with button_col:
    generate = st.button("Generate lesson", type="primary", use_container_width=True)

if generate:
    if not api_key:
        st.error(f"Set {env_key_name} in your environment or Streamlit secrets to use {provider}.")
    elif not topic.strip():
        st.error("Enter a topic to get started.")
    else:
        try:
            status = st.status("Starting lesson generation…", expanded=True)

            def report_progress(message: str) -> None:
                status.write(message)

            with status:
                run = run_pipeline(
                    provider=provider,
                    model=model.strip(),
                    api_key=api_key,
                    topic=topic.strip(),
                    max_retries=max_retries,
                    adaptive_guidance=memory.adaptive_guidance(),
                    progress_callback=report_progress,
                )
                memory.save_run(run)
                status.update(
                    label="Lesson ready" if run["passed"] else "Lesson did not pass the rubric",
                    state="complete" if run["passed"] else "error",
                    expanded=not run["passed"],
                )
            st.session_state["active_run_id"] = run["id"]
        except Exception as error:
            if "status" in locals():
                status.update(label="Failed to generate lesson", state="error", expanded=True)
            st.error(f"The run could not be completed: {error}")

recent = memory.recent_runs()
if recent:
    st.divider()
    labels = {
        run["id"]: f"{'PASS' if run['passed'] else 'REVIEW'} · {run['topic']} · {run['created_at'][:16].replace('T', ' ')}"
        for run in recent
    }
    active_id = st.session_state.get("active_run_id")
    default_index = next((index for index, item in enumerate(recent) if item["id"] == active_id), 0)
    chosen_id = st.selectbox(
        "Run history",
        options=[run["id"] for run in recent],
        index=default_index,
        format_func=lambda run_id: labels[run_id],
    )
    chosen_run = memory.get_run(chosen_id)
    if chosen_run:
        render_run(chosen_run)
        with st.expander("Add human feedback to persistent memory"):
            with st.form(f"feedback-{chosen_id}"):
                rating = st.radio("How useful was this result?", ["Helpful", "Needs work"], horizontal=True)
                note = st.text_area("What should future lessons do differently?", placeholder="Mention clarity, accuracy, examples, or anything the rubric missed.")
                submitted = st.form_submit_button("Save feedback")
            if submitted:
                if note.strip():
                    memory.add_feedback(chosen_id, rating, note)
                    st.success("Feedback saved for future runs.")
                else:
                    st.warning("Add a short note before saving feedback.")
            for item in memory.feedback_for_run(chosen_id):
                st.caption(f"{item['rating']} · {item['note']}")
else:
    st.divider()
    st.markdown("#### Quality gate")
    st.write("Accuracy and grounding · beginner language · worked example · explained terms · complete coverage · coherent flow")
    st.caption("The evaluator sees the draft and rubric, but not the generator's reasoning. Every checkpoint must pass; otherwise the next draft receives the exact failure reasons.")