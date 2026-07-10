

import tempfile
import os
import io
import uuid
#import requests
from typing import Annotated, Dict, Any
from predictor import predict_image
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()



PROJECT_DIR = r"C:\Users\shash\OneDrive\Desktop\Primetals_Work\Defect_Classifier\SteelDefectAI"


_DEFECT_PROFILES = {
    "Crazing": dict(sem="Fine network of intergranular micro-cracks at surface",
                     ut="Weak, scattered backscatter near surface (<2mm depth)",
                     eds="Elevated O and slight S segregation at grain boundaries",
                     ir="Localized hot spots during rolling, ~15-25°C above mean"),
    "Inclusion": dict(sem="Angular non-metallic particles embedded in matrix",
                       ut="Discrete point reflector, mid-thickness",
                       eds="Al2O3 / MnS composition signature at particle site",
                       ir="No strong thermal anomaly (subsurface defect)"),
    "Patches": dict(sem="Irregular scale patches with underlying pitting",
                     ut="Minor surface-only attenuation",
                     eds="High Fe-oxide (FeO/Fe3O4) concentration",
                     ir="Uneven descaling — cold patches vs surrounding strip"),
    "Pitted Surface": dict(sem="Discrete pits with rounded edges, ~10-50 micron",
                            ut="Surface-breaking indications, shallow",
                            eds="Localized Cl or S enrichment (pitting corrosion signature)",
                            ir="Minor localized cooling irregularities"),
    "Rolled-in Scale": dict(sem="Flattened oxide layer pressed into surface",
                             ut="Laminar-type reflector near surface",
                             eds="Thick FeO scale layer, poor descaling adhesion break",
                             ir="Mild thermal shadow where scale insulated the surface"),
    "Scratches": dict(sem="Linear grooves, uniform depth, rolling direction aligned",
                       ut="No significant subsurface indication",
                       eds="No compositional anomaly (mechanical origin)",
                       ir="No thermal anomaly (mechanical, not process-thermal)"),
}

BASE_SYSTEM_PROMPT = """You are "MetSight", a senior steelmaking process metallurgist \
and root-cause-analysis copilot embedded in a digital-twin system for a hot \
strip / continuous casting line.

You are given:
- A defect class predicted by a computer-vision model (ResNet50) from a surface image.
- REAL process parameters entered by the plant operator, covering both general
  casting/rolling conditions (steel grade, casting speed, mold/finish-rolling/
  coiling temperatures, descaling pressure) AND detailed mold-oscillation /
  casting-powder parameters (mold stroke, oscillation frequency, negative/positive
  strip time, total cycle time, negative strip ratio, oscillation depth, mold
  oscillation index, casting powder viscosity, powder consumption, and the
  oscillation waveform modification ratio (Arccos-based, -1 to +1)).

CRITICAL DATA-INTEGRITY RULE: The process and mold-oscillation parameter values
given to you in the user's message are REAL, operator-entered measurements.
You must use those exact values in your reasoning and quote them exactly as
given. NEVER invent, estimate, guess-fill, or substitute different numbers for
them, and NEVER treat any tool output as a source of process parameters --
tools in this system only ever return supplementary synthetic inspection
readings (SEM/UT/EDS/IR), not process data. If a value you need was not
provided, say explicitly that it is missing rather than fabricating one.

- Mold-oscillation and casting-powder parameters are especially relevant when
  reasoning about surface-related defects (e.g. Crazing, Patches, Rolled-in
  Scale, Pitted Surface): poor lubrication (low powder consumption / high
  viscosity), an insufficient negative strip ratio, or an atypical oscillation
  waveform can each independently explain such defects, so weigh them alongside
  the thermal/casting-speed parameters rather than defaulting only to the
  latter.
{tool_desc}

Using a ReAct approach (reason -> call a tool if you need evidence -> observe -> repeat):
1. Call `generate_ndt_readings` for the predicted defect class to get supporting inspection data.
{search_step}{final_step_number}. Then produce a structured final answer with these sections:
   - **Root Cause** (most likely mechanism, stated plainly)
   - **Supporting Evidence** (tie together image classification, SEM/UT/EDS/IR, process parameters)
   - **Explanation** (why the process conditions produced this defect, referencing metallurgical theory)
   - **Recommendations** (concrete corrective actions for the plant, prioritized)
   - **Confidence & Caveats** (note this is a POC with synthetic inspection data where relevant)

Be concise, technically precise, and avoid inventing citations. If a tool returns nothing
useful, say so and reason from first-principles metallurgy instead. NEVER call a tool
that is not explicitly listed above as available to you.
When asked who created you tell Shashwata Roy , Intern of Primetals Technologies India created me 
"""


def build_system_prompt(tool_names: list) -> str:
    """Builds the system prompt so it only ever references tools that are
    actually bound to the LLM for this run -- prevents the model from trying
    to call a tool that wasn't included in the request (which Groq rejects
    with a 400 tool_use_failed error)."""
    has_web = "steel_web_search" in tool_names

    if has_web:
        tool_desc = (
            "- A tool to fetch synthetic SEM / UT / EDS / IR inspection readings "
            "correlated with the defect (NOT process parameters — those are given "
            "to you directly and are real), and a tool to search the live internet "
            "for metallurgical theory, defect mechanisms, standards, papers, or "
            "recent practice notes on the relevant process parameters."
        )
        search_step = (
            "2. Call `steel_web_search` to look up the metallurgical mechanism, typical "
            "process-parameter ranges, or relevant standards/practice notes for the "
            "predicted defect -- search the internet rather than relying on memory.\n"
        )
        final_step_number = "3"
    else:
        tool_desc = (
            "- A tool to fetch synthetic SEM / UT / EDS / IR inspection readings "
            "correlated with the defect (NOT process parameters — those are given "
            "to you directly and are real). (Web search is currently disabled -- "
            "reason from first-principles metallurgical theory instead.)"
        )
        search_step = ""
        final_step_number = "2"

    return BASE_SYSTEM_PROMPT.format(
        tool_desc=tool_desc, search_step=search_step, final_step_number=final_step_number
    )


@tool
def generate_ndt_readings(defect_class: str) -> Dict[str, Any]:
    """Generate POC-scope synthetic SEM, UT, EDS, and IR inspection readings
    correlated with a given steel surface defect class.
    Use this AFTER you know the predicted defect class, to obtain supporting
    (synthetic, illustrative) inspection evidence for root-cause reasoning.

    IMPORTANT: This tool does NOT and must NOT be used as a source of process
    parameters (casting speed, mold temperature, mold oscillation, casting
    powder, etc.). Those are real values entered by the plant operator and
    are already present in the conversation -- always reason from those real
    values, never from anything generated here.

    Args:
        defect_class: one of Crazing, Inclusion, Patches, Pitted Surface,
            Rolled-in Scale, Scratches (case-insensitive, partial match ok).
    """
    match = next(
        (k for k in _DEFECT_PROFILES if k.lower() in defect_class.lower()
         or defect_class.lower() in k.lower()),
        None,
    )
    profile = _DEFECT_PROFILES.get(match, {
        "sem": "No characteristic microstructure profile available",
        "ut": "No characteristic UT signature available",
        "eds": "No characteristic EDS profile available",
        "ir": "No characteristic thermal signature available",
    })

    return {
        "defect_class": match or defect_class,
        "sem_analysis": profile.get("sem"),
        "ultrasonic_testing": profile.get("ut"),
        "eds_analysis": profile.get("eds"),
        "infrared_thermography": profile.get("ir"),
        "disclaimer": (
            "Synthetic POC inspection data — illustrative only, not measured. "
            "Process and mold-oscillation parameters are NOT included here; "
            "use the real operator-entered values from the conversation."
        ),
    }


def extract_top_k(pred: Dict[str, Any], k: int = 3):
    """Extract the top-k (class, confidence) pairs from predict_image()'s
    return value.

    predictor.py returns pred["top3"] as a list of
    {"class": ..., "probability": ...} dicts (already sorted, argsort
    descending). A few alternate key-name conventions are also supported as
    a fallback in case predictor.py changes shape later. Returns None only
    if none of these are present, so callers can degrade gracefully.
    """
    for key in ("top3", "top_k", "topk", "top_predictions", "all_scores",
                "class_probabilities", "probabilities"):
        val = pred.get(key)
        if not val:
            continue

        items = []
        if isinstance(val, dict):
            items = list(val.items())
        elif isinstance(val, (list, tuple)):
            for entry in val:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    items.append((entry[0], entry[1]))
                elif isinstance(entry, dict) and "class" in entry:
                    score = entry.get("probability", entry.get("confidence", entry.get("score", 0)))
                    items.append((entry["class"], score))

        if items:
            items.sort(key=lambda kv: kv[1], reverse=True)
            return items[:k]

    return None


def load_web_search_tool():
    """Live internet search tool (replaces the old book-RAG retriever).
    Used by the agent to look up metallurgical mechanisms, typical process
    parameter ranges, and standards/practice notes instead of a static PDF."""
    if not os.getenv("TAVILY_API_KEY"):
        return None
    tavily = TavilySearch(max_results=2, topic="general", include_raw_content=False)
    # Give it the exact name the SYSTEM_PROMPT refers to, and a description
    # that steers it toward steelmaking process-parameter lookups.
    tavily.name = "steel_web_search"
    tavily.description = (
        "Search the live internet for steelmaking metallurgical theory, defect "
        "mechanisms, typical/standard process-parameter ranges (casting speed, "
        "mold temperature, finish rolling temperature, coiling temperature, "
        "descaling pressure, etc.), industry standards, or recent practice notes. "
        "Use this instead of relying on memory whenever you need grounding for "
        "process parameters or defect mechanisms."
    )
    return tavily


def build_toolset():
    tools = [generate_ndt_readings]

    web_tool = load_web_search_tool()
    if web_tool:
        tools.append(web_tool)

    return tools

class State(TypedDict):
    messages: Annotated[list, add_messages]


@st.cache_resource(show_spinner=False)
def build_agent():
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_PROVIDER = "groq"

    llm = init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER, temperature=0.2)

    tools = build_toolset()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    tool_names = [t.name for t in tools]
    system_prompt = build_system_prompt(tool_names)


    MAX_HISTORY_MESSAGES = 10

    def tool_calling_llm(state: State):
        messages = state["messages"]
        if messages and isinstance(messages[0], SystemMessage):
            messages = messages[1:]
        trimmed = list(messages)[-MAX_HISTORY_MESSAGES:]
        messages = [SystemMessage(content=system_prompt)] + trimmed
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges("tool_calling_llm", tools_condition)
        builder.add_edge("tools", "tool_calling_llm")
    else:
        builder.add_edge(START, "tool_calling_llm")
        builder.add_edge("tool_calling_llm", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)




st.set_page_config(
    page_title="MetSight | Steel Defect Digital Twin",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    :root {
        --ms-accent: #ff6a3d;
        --ms-accent-dark: #e0532a;
        --ms-bg-card: #1a1410;
        --ms-border: #3a2b20;
    }

    /* ---- Keyframes ---- */
    @keyframes ms-gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes ms-glow-pulse {
        0%, 100% { box-shadow: 0 0 12px rgba(255,106,61,0.25), 0 0 0 1px rgba(255,106,61,0.25) inset; }
        50%      { box-shadow: 0 0 28px rgba(255,106,61,0.55), 0 0 0 1px rgba(255,106,61,0.45) inset; }
    }
    @keyframes ms-icon-float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50%      { transform: translateY(-4px) rotate(-3deg); }
    }
    @keyframes ms-shimmer {
        0%   { transform: translateX(-120%) rotate(20deg); }
        100% { transform: translateX(220%) rotate(20deg); }
    }
    @keyframes ms-dot-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(110,231,183,0.55); }
        70%  { box-shadow: 0 0 0 8px rgba(110,231,183,0); }
        100% { box-shadow: 0 0 0 0 rgba(110,231,183,0); }
    }
    @keyframes ms-fade-up {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---- App-wide ---- */
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(255,106,61,0.14) 0%, transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(255,138,91,0.07) 0%, transparent 45%),
            linear-gradient(120deg, #17120d 0%, #1f1712 50%, #17120d 100%);
        background-size: 200% 200%, 200% 200%, 200% 200%;
        animation: ms-gradient-shift 18s ease infinite;
    }
    .block-container { padding-top: 1.6rem; max-width: 1250px; }
    #MainMenu, footer { visibility: hidden; }

    /* ---- Header banner ---- */
    .ms-header {
        position: relative; overflow: hidden;
        display: flex; align-items: center; gap: 18px;
        padding: 24px 30px; border-radius: 18px; margin-bottom: 24px;
        background: linear-gradient(120deg, #1b1008 0%, #201513 45%, #14171c 100%);
        background-size: 200% 200%;
        animation: ms-gradient-shift 10s ease infinite, ms-glow-pulse 4s ease-in-out infinite;
        border: 1px solid rgba(255,106,61,0.3);
    }
    .ms-header::before {
        content: ""; position: absolute; top: -60%; left: -10%;
        width: 40%; height: 220%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.07), transparent);
        animation: ms-shimmer 6s linear infinite;
    }
    .ms-header .ms-icon {
        font-size: 2.6rem; line-height: 1; position: relative; z-index: 1;
        filter: drop-shadow(0 0 14px rgba(255,106,61,0.55));
        animation: ms-icon-float 3.5s ease-in-out infinite;
    }
    .ms-header h1 {
        font-size: 1.75rem; margin: 0; font-weight: 800; letter-spacing: 0.2px;
        background: linear-gradient(90deg, #ffffff 0%, #ffd9c4 45%, #ff8a5b 70%, #ffffff 100%);
        background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: ms-gradient-shift 5s linear infinite;
    }
    .ms-header .ms-tag {
        font-weight: 800;
        background: linear-gradient(135deg, var(--ms-accent), #ffb37a);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .ms-header p { margin: 4px 0 0 0; color: #9aa3af; font-size: 0.92rem; position: relative; z-index: 1; }

    /* ---- Step badges ---- */
    .ms-step {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(255,106,61,0.10);
        border: 1px solid rgba(255,106,61,0.4);
        color: #ffb08a; font-weight: 700; font-size: 0.78rem;
        padding: 5px 14px; border-radius: 999px; margin-bottom: 12px;
        text-transform: uppercase; letter-spacing: 0.8px;
        animation: ms-fade-up 0.5s ease both;
    }
    .ms-step::before {
        content: ""; width: 6px; height: 6px; border-radius: 50%;
        background: var(--ms-accent);
        box-shadow: 0 0 8px 2px rgba(255,106,61,0.7);
    }

    /*
       ---- Section "cards" ----
       IMPORTANT: We can't wrap live Streamlit widgets in a hand-written
       <div class="ms-card"> ... </div> split across multiple st.markdown()
       calls -- each st.markdown() call creates its own isolated DOM node,
       so the opening and closing tags never actually nest the widgets
       between them. Instead, each card puts an invisible <span> "marker"
       inside a real `st.container()`, and we style the ANCESTOR Streamlit
       block that contains that marker using the :has() selector. This way
       the real container (which truly wraps all the widgets) gets the
       card look, with no orphaned/floating boxes.
    */
    div[data-testid="stVerticalBlock"]:has(> div > div > span.ms-card-marker) {
        background: linear-gradient(160deg, var(--ms-bg-card) 0%, #0e1116 100%);
        border: 1px solid var(--ms-border);
        border-radius: 16px;
        padding: 20px 22px;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: ms-fade-up 0.5s ease both;
    }
    div[data-testid="stVerticalBlock"]:has(> div > div > span.ms-card-marker):hover {
        transform: translateY(-3px);
        border-color: rgba(255,106,61,0.45);
        box-shadow: 0 14px 34px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,106,61,0.15);
    }
    .ms-card h3 { margin-top: 0; font-size: 1.05rem; color: #eceef1; }
    .ms-subtle { color: #7d8590; font-size: 0.85rem; }

    /* ---- Metric-style result pills ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #211a14, #17120d);
        border: 1px solid var(--ms-border); border-radius: 14px;
        padding: 14px 16px 10px 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: ms-fade-up 0.5s ease both;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(255,106,61,0.15);
    }
    div[data-testid="stMetricLabel"] { color: #9aa3af !important; }
    div[data-testid="stMetricValue"] {
        color: #ffb08a !important;
        text-shadow: 0 0 18px rgba(255,106,61,0.45);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 10px !important; font-weight: 700 !important;
        border: 1px solid var(--ms-border) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease !important;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    .stButton > button[kind="primary"] {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, var(--ms-accent), var(--ms-accent-dark)) !important;
        background-size: 200% 200% !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(255,106,61,0.4);
        animation: ms-gradient-shift 3s ease infinite;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.12);
        box-shadow: 0 8px 26px rgba(255,106,61,0.55);
    }

    /* ---- Progress bar glow ---- */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--ms-accent), #ffcf9e) !important;
        box-shadow: 0 0 10px rgba(255,106,61,0.6);
    }

    /* ---- Expander / trace ---- */
    div[data-testid="stExpander"] {
        border-radius: 14px !important; border: 1px solid var(--ms-border) !important;
        background: #16110d !important;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stExpander"]:hover { border-color: rgba(255,106,61,0.35) !important; }

    /* ---- Answer box ---- */
    .ms-answer {
        background: linear-gradient(160deg, var(--ms-bg-card) 0%, #0e1116 100%);
        border: 1px solid rgba(255,106,61,0.3);
        border-radius: 16px; padding: 24px 28px; margin-top: 6px;
        box-shadow: 0 0 0 1px rgba(255,106,61,0.08), 0 10px 40px rgba(0,0,0,0.45);
        animation: ms-fade-up 0.6s ease both, ms-glow-pulse 5s ease-in-out infinite;
    }

    /* ---- Divider spacing ---- */
    hr { margin: 1.4rem 0 !important; border-color: var(--ms-border) !important; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #150f0b 0%, #100b08 100%);
        border-right: 1px solid var(--ms-border);
    }
    .ms-status-ok, .ms-status-warn {
        display: inline-flex; align-items: center; gap: 8px; font-weight: 700;
    }
    .ms-status-ok { color: #6ee7b7; }
    .ms-status-warn { color: #fbbf24; }
    .ms-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #6ee7b7; animation: ms-dot-pulse 1.8s infinite;
    }
    .ms-dot-warn { background: #fbbf24; }
</style>
""", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "prediction" not in st.session_state:
    st.session_state.prediction = None

agent = build_agent()


st.markdown("""
<div class="ms-header">
    <div class="ms-icon">🏭</div>
    <div>
        <h1>MetSight <span class="ms-tag">v2.0</span></h1>
        <p>⚡  Defect Root-Cause Copilot — Steelmaking · Hot Strip / Continuous Casting</p>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="ms-step">Step 1 · Inputs</div>', unsafe_allow_html=True)

col_upload, col_params = st.columns(2, gap="large")

with col_upload:
    with st.container():
        # Invisible marker: lets the CSS above style THIS real container
        # (the one that actually wraps every widget below) as a card.
        st.markdown('<span class="ms-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("###  Upload Defect Image")
        st.markdown('<p class="ms-subtle">Surface defect image from steel strip or plate inspection.</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Surface defect image (steel strip / plate)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image_bytes = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            img = Image.open(uploaded_file)
            preprocessed = img.convert("RGB").resize((224, 224))
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(img, caption="Original", use_container_width=True)
            with img_col2:
                st.image(preprocessed, caption="224×224 (ResNet50 input)", use_container_width=True)
        else:
            st.info("No image uploaded yet — drop a JPG/PNG above to get started.", icon="🖼️")

with col_params:
    with st.container():
        st.markdown('<span class="ms-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("###  Process Parameters")
        st.markdown('<p class="ms-subtle">Operator-entered casting & rolling conditions for this coil.</p>', unsafe_allow_html=True)

        steel_grade = st.selectbox(
            "Steel Grade",
            ["IF Steel (Ti-stabilized)", "Low-C DQSK", "HSLA-350", "Plain Carbon 1010", "Other"],
        )

        p1, p2 = st.columns(2)
        with p1:
            # Concast spec range: 1-7 m/min
            casting_speed = st.number_input("Casting Speed (m/min)", 0.5, 8.0, 3.0, 0.1)
            finish_roll_temp = st.number_input("Finish Rolling Temp (°C)", 750.0, 950.0, 860.0, 1.0)
            descaling_pressure = st.number_input("Descaling Pressure (bar)", 100.0, 300.0, 180.0, 1.0)
        with p2:
            mold_temp = st.number_input("Mold Temperature (°C)", 1450.0, 1650.0, 1545.0, 1.0)
            coiling_temp = st.number_input("Coiling Temperature (°C)", 450.0, 750.0, 600.0, 1.0)

        operator_notes = st.text_area(
            "Operator Notes (optional)",
            placeholder="Anything unusual observed on the line...",
            height=90,
        )


with st.container():
    st.markdown('<span class="ms-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("###  Mold Oscillation & Casting Powder Parameters")
    st.markdown(
        '<p class="ms-subtle">Concast-specific process variables — passed to the '
        'root-cause agent alongside the parameters above.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Show / edit oscillation & casting-powder parameters", expanded=False):
        o1, o2, o3 = st.columns(3)
        with o1:
            mold_stroke_mm = st.number_input(
                "Mold Stroke (mm)", 3.0, 15.0, 7.5, 0.1,
                help="Linear movement of the mold. Typical range: 5-10 mm.",
            )
            oscillation_freq_per_min = st.number_input(
                "Oscillation Frequency (cycles/min)", 100.0, 250.0, 175.0, 1.0,
                help="Frequency of the mold stroke. Typical range: 150-200/min.",
            )
            negative_strip_time_sec = st.number_input(
                "Negative Strip Time (sec)", 0.05, 0.3, 0.15, 0.01,
                help="Time the mold moves faster than the strand (downward). Typical range: 0.1-0.2 sec.",
            )
            oscillation_depth_mm = st.number_input(
                "Oscillation Depth (mm)", 0.05, 0.30, 0.14, 0.01,
                help="Depth of oscillation. Typical range: 0.11-0.18 mm.",
            )
        with o2:
            total_cycle_time_sec = st.number_input(
                "Total Cycle Time (sec)", 0.20, 0.50, 0.35, 0.01,
                help="Full oscillation cycle duration. Typical range: 0.3-0.4 sec.",
            )
            positive_strip_time_pct = st.number_input(
                "Positive Strip Time (% of cycle)", 5.0, 50.0, 20.0, 0.5,
                help="Positive strip time as a percentage of total cycle time. Typical range: ~15-30%.",
            )
            mold_oscillation_index = st.number_input(
                "Mold Oscillation Index", 0.015, 1.2, 0.5, 0.005,
                help="Non-dimensional mold oscillation index. Typical range: 0.015 - 1.0.",
            )
            arccos_number = st.slider(
                "Oscillation Waveform Modification Ratio (Arccos-based)", -1.0, 1.0, 0.0, 0.01,
                help="Describes deviation from a pure sinusoidal oscillation waveform. "
                     "0 = pure sinusoidal. Range: -1 to +1.",
            )
        with o3:
            casting_powder_viscosity_pas = st.number_input(
                "Casting Powder Viscosity (Pa·s)", 0.05, 0.30, 0.15, 0.01,
                help="Mold flux viscosity. Typical range: 0.10-0.20 Pa·s.",
            )
            powder_consumption_kg_m2 = st.number_input(
                "Powder Consumption (kg/m² strand area)", 0.10, 0.50, 0.28, 0.01,
                help="Mold flux consumption per unit strand area. Typical range: 0.17-0.40 kg/m².",
            )
            negative_strip_ratio_pct = st.number_input(
                "Negative Strip Ratio (% of total cycle time)", 0.0, 100.0, 30.0, 1.0,
                help="Negative strip time as a percentage of total cycle time.",
            )
            computed_ratio = (
                (negative_strip_time_sec / total_cycle_time_sec * 100)
                if total_cycle_time_sec else 0.0
            )
            st.caption(
                f"ℹ️ Negative strip time ÷ total cycle time = {computed_ratio:.1f}% "
                "— compare against the value entered above."
            )

mold_oscillation_params = {
    "mold_stroke_mm": mold_stroke_mm,
    "oscillation_frequency_per_min": oscillation_freq_per_min,
    "casting_speed_m_min": casting_speed,
    "negative_strip_time_sec": negative_strip_time_sec,
    "total_cycle_time_sec": total_cycle_time_sec,
    "positive_strip_time_pct_of_cycle": positive_strip_time_pct,
    "oscillation_depth_mm": oscillation_depth_mm,
    "mold_oscillation_index": mold_oscillation_index,
    "casting_powder_viscosity_Pa_s": casting_powder_viscosity_pas,
    "powder_consumption_kg_per_m2": powder_consumption_kg_m2,
    "negative_strip_ratio_pct": negative_strip_ratio_pct,
    "oscillation_waveform_arccos_number": arccos_number,
}

user_process_params = {
    "steel_grade": steel_grade,
    "casting_speed_m_min": casting_speed,
    "mold_temperature_C": mold_temp,
    "finish_rolling_temp_C": finish_roll_temp,
    "coiling_temp_C": coiling_temp,
    "descaling_pressure_bar": descaling_pressure,
    "operator_notes": operator_notes or "None",
}

st.write("")
run_classification = st.button(
    "🔍 Run Classification Engine",
    type="primary",
    disabled=(image_bytes is None),
    use_container_width=True,
)

if run_classification and image_bytes is not None:
    with st.spinner("Running Hyper Classification Engine ..."):
        image = Image.open(io.BytesIO(image_bytes))
        st.session_state.prediction = predict_image(image)
     
        st.session_state.thread_id = str(uuid.uuid4())

if st.session_state.prediction:
    pred = st.session_state.prediction
    confidence = pred["confidence"]
    st.markdown('<div class="ms-step">Step 2 · Classification Result</div>', unsafe_allow_html=True)

    if confidence < 0.40:
        st.error(
            f"🚫 **Very low confidence — unclassified defect / error finding the defect.** "
            f"(Best guess was *{pred['predicted_class']}* at {confidence*100:.1f}%, below the "
            "40% reliability threshold.) Consider re-capturing the image (lighting, focus, "
            "angle) or inspecting the coil manually before proceeding.",
        )
    elif confidence < 0.50:
        st.warning(
            f"⚠️ Low confidence ({confidence*100:.1f}%) — showing the top 3 possibilities "
            "instead of a single call.",
        )
        top3 = extract_top_k(pred, k=3)
        if top3:
            cols = st.columns(len(top3))
            for col, (cls, score) in zip(cols, top3):
                col.metric(cls, f"{score*100:.1f}%")
        else:
            st.info(
                f"Top guess: **{pred['predicted_class']}** ({confidence*100:.1f}%). Full "
                "top-3 breakdown isn't available because `predict_image()` currently only "
                "returns the single top prediction — add a `top_k` (or `probabilities`) "
                "field to its return value to enable this.",
            )
    else:
        c1, c2, c3 = st.columns([1, 1, 1.4])
        c1.metric("Predicted Defect Class", pred["predicted_class"])
        c2.metric("Confidence", f"{confidence*100:.1f}%")
        with c3:
            st.progress(min(max(confidence, 0.0), 1.0))
            if pred.get("note"):
                st.caption(f"ℹ️ {pred['note']}")

st.divider()


st.markdown('<div class="ms-step">Step 3 · Root-Cause Reasoning</div>', unsafe_allow_html=True)
st.markdown("###  Run Root Cause Analysis Engine")
st.markdown(
    '<p class="ms-subtle">The ReAct agent pulls synthetic NDT/process readings'
    + (' and live web evidence' if os.getenv("TAVILY_API_KEY") else '')
    + ' before producing a structured root-cause report.</p>',
    unsafe_allow_html=True,
)
_pred_unclassified = (
    st.session_state.prediction is not None
    and st.session_state.prediction["confidence"] < 0.40
)
run_analysis = st.button(
    "▶ Run Root Cause Analysis",
    disabled=(st.session_state.prediction is None or _pred_unclassified),
    use_container_width=True,
)
if _pred_unclassified:
    st.caption(
        "🚫 Root-cause analysis is disabled — the last classification was below the "
        "40% confidence threshold and treated as unclassified. Re-capture the image and "
        "re-run classification first."
    )

if run_analysis:
    pred = st.session_state.prediction
    confidence = pred["confidence"]

    classification_note = ""
    if confidence < 0.50:
        top3 = extract_top_k(pred, k=3)
        if top3:
            top3_lines = os.linesep.join(f"  - {cls}: {score*100:.1f}%" for cls, score in top3)
            classification_note = f"""
NOTE: Classification confidence is below 50% ({confidence*100:.1f}%), so this is
AMBIGUOUS between several defect classes, not a confident single call. Top 3
candidates:
{top3_lines}
Reason about the most metallurgically plausible candidate given the process
parameters below, and explicitly state in your answer that the visual
classification itself was low-confidence/ambiguous.
"""
        else:
            classification_note = f"""
NOTE: Classification confidence is below 50% ({confidence*100:.1f}%) — treat the
predicted class as a tentative best guess only, and explicitly flag this
uncertainty in your answer rather than treating it as a confirmed defect class.
"""

    user_prompt = f"""A digital-twin surface inspection has produced the following. \
Please run your ReAct workflow ( \
search the web) and give me the root cause analysis.

Predicted defect class: {pred['predicted_class']}
Classification confidence: {confidence*100:.1f}%
{classification_note}
Operator-entered process parameters:
{os.linesep.join(f"- {k}: {v}" for k, v in user_process_params.items())}

Mold oscillation & casting-powder parameters (concast process variables):
{os.linesep.join(f"- {k}: {v}" for k, v in mold_oscillation_params.items())}
"""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    trace_placeholder = st.expander("🔧 Agent tool-call trace (ReAct steps)", expanded=False)
    final_answer_box = st.empty()

    with st.spinner("Agent reasoning (tool calls in progress)..."):
        events = agent.stream(
            {"messages": [HumanMessage(content=user_prompt)]},
            config=config, stream_mode="values",
        )
        last_ai_content = None
        for event in events:
            msg = event["messages"][-1]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    trace_placeholder.markdown(f"**Calling tool:** `{tc['name']}`  \nArgs: `{tc['args']}`")
            elif isinstance(msg, ToolMessage):
                trace_placeholder.markdown(f"**Tool `{msg.name}` returned:**")
                trace_placeholder.code(str(msg.content)[:1500])
            elif isinstance(msg, AIMessage) and msg.content:
                last_ai_content = msg.content

    if last_ai_content:
   
        final_answer_box.markdown(
            f'<div class="ms-answer">\n\n{last_ai_content}\n\n</div>',
            unsafe_allow_html=True,
        )

st.divider()


st.markdown('<div class="ms-step">Step 4 · Follow-up</div>', unsafe_allow_html=True)
st.markdown("### 💬 Ask a follow-up question")
follow_up = st.chat_input("e.g. What if we lower the descaling pressure by 20 bar?")
if follow_up:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Thinking..."):
        result = agent.invoke({"messages": [HumanMessage(content=follow_up)]}, config=config)
    st.chat_message("user").write(follow_up)
    st.chat_message("assistant").write(result["messages"][-1].content)

with st.sidebar:
    st.markdown("##  MetSight v2.0")
    st.caption("Digital-twin control panel")
    st.divider()

    st.markdown("**System Status**")
    st.markdown('<span class="ms-status-ok"><span class="ms-dot"></span> ResNet50 Visual Inspection Model Loaded</span>', unsafe_allow_html=True)
    if os.getenv("TAVILY_API_KEY"):
        st.markdown('<span class="ms-status-ok"><span class="ms-dot"></span> Tavily Search Enabled</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="ms-status-warn"><span class="ms-dot ms-dot-warn"></span> Tavily Disabled</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Session**")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")
    if st.session_state.prediction:
        st.caption(f"Last prediction: **{st.session_state.prediction['predicted_class']}**")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.prediction = None
        st.rerun()

    st.divider()
    st.caption("MetSight v2.0 ·")