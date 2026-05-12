import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import json
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – dark industrial dashboard look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141821 !important;
    border-right: 1px solid #2d3748;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px;
}

/* Code blocks */
.stCode, code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #111827 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 6px;
}

/* Buttons */
.stButton > button {
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.45rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2563eb; }

/* Text input */
.stTextInput > div > div > input {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 6px;
    color: #e2e8f0;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dataframe */
.stDataFrame { border: 1px solid #2d3748; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #1a202c;
    border-radius: 6px 6px 0 0;
    color: #94a3b8;
    font-weight: 600;
    border: 1px solid #2d3748;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: #fff !important;
}

/* History items */
.history-item {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    font-size: 0.85rem;
    color: #94a3b8;
}
.history-item:hover { border-color: #3b82f6; color: #e2e8f0; }

/* Section headers */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #3b82f6 !important; letter-spacing: -1px; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e2e8f0 !important; }

/* Divider */
hr { border-color: #2d3748; }

/* Select box */
.stSelectbox > div { background: #1a202c; border: 1px solid #2d3748; border-radius: 6px; }

/* Success / Error */
.stSuccess { background: #064e3b !important; border: 1px solid #059669 !important; }
.stError   { background: #450a0a !important; border: 1px solid #dc2626 !important; }

/* API key badge */
.api-badge-ok  { color: #10b981; font-size: 0.78rem; font-weight: 600; }
.api-badge-err { color: #f87171; font-size: 0.78rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("query_history", []),
    ("last_result_df", None),
    ("last_sql", ""),
    ("last_fig", None),
    ("df", None),
    ("table_name", "data"),
    ("groq_api_key", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  LLM  – created on demand so key changes apply
# ─────────────────────────────────────────────
def get_llm(api_key: str):
    # from langchain_groq import ChatGroq
    return ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile",
    )

# ─────────────────────────────────────────────
#  SQLITE HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return sqlite3.connect(":memory:", check_same_thread=False)

conn = get_conn()


def load_df_to_sqlite(df: pd.DataFrame, table: str = "data"):
    df.to_sql(table, conn, if_exists="replace", index=False)


def run_sql(query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)


def get_schema(df: pd.DataFrame, table: str = "data") -> str:
    cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        sample = df[col].dropna().head(3).tolist()
        cols.append(f"  {col} ({dtype})  -- e.g. {sample}")
    return f"Table: {table}\nColumns:\n" + "\n".join(cols)

# ─────────────────────────────────────────────
#  PROMPT BUILDERS
# ─────────────────────────────────────────────
SQL_SYSTEM = """You are an expert SQL data analyst.
Given a SQLite table schema and a user question, respond ONLY with a valid SQLite SELECT query.
Rules:
- No markdown fences, no explanation, no comments.
- Output raw SQL only.
- Always alias aggregations (e.g. COUNT(*) AS count).
- Use the exact table name provided.
- Limit results to 500 rows unless the user asks for everything.
"""

VIZ_SYSTEM = """You are a data visualization expert.
Given a question and the column names + dtypes of a result dataframe, respond ONLY with a JSON object.
JSON fields:
  chart_type: one of bar | line | scatter | pie | histogram | box | heatmap | none
  x: column name for x-axis (or null)
  y: column name for y-axis (or null)
  color: column name for color grouping (or null)
  title: a short chart title
  insight: one-sentence insight about what the chart shows

Respond with raw JSON only. No markdown, no explanation.
"""


def generate_sql(question: str, schema: str, api_key: str) -> str:
    llm = get_llm(api_key)
    prompt = f"{SQL_SYSTEM}\n\nSchema:\n{schema}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    sql = response.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql


def generate_viz_spec(question: str, result_df: pd.DataFrame, api_key: str) -> dict:
    llm = get_llm(api_key)
    col_info = ", ".join(
        f"{c} ({t})" for c, t in zip(result_df.columns, result_df.dtypes)
    )
    prompt = f"{VIZ_SYSTEM}\n\nQuestion: {question}\nResult columns: {col_info}\nRow count: {len(result_df)}"
    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"chart_type": "none", "title": "", "insight": ""}


# ─────────────────────────────────────────────
#  CHART BUILDER
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"


def build_chart(spec: dict, df: pd.DataFrame):
    ct = spec.get("chart_type", "none")
    x, y, color, title = spec.get("x"), spec.get("y"), spec.get("color"), spec.get("title", "")

    def col(c):
        return c if c and c in df.columns else None

    x, y, color = col(x), col(y), col(color)

    try:
        if ct == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE, text_auto=True)
        elif ct == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE, markers=True)
        elif ct == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE)
        elif ct == "pie":
            fig = px.pie(df, names=x, values=y, title=title, template=PLOTLY_TEMPLATE)
        elif ct == "histogram":
            fig = px.histogram(df, x=x, color=color, title=title, template=PLOTLY_TEMPLATE)
        elif ct == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title, template=PLOTLY_TEMPLATE)
        elif ct == "heatmap":
            pivot = df.pivot_table(index=x, columns=color, values=y, aggfunc="sum") if color else None
            if pivot is not None:
                fig = px.imshow(pivot, title=title, template=PLOTLY_TEMPLATE)
            else:
                return None
        else:
            return None

        fig.update_layout(
            plot_bgcolor="#141821",
            paper_bgcolor="#141821",
            font=dict(family="IBM Plex Sans", color="#e2e8f0"),
            margin=dict(t=48, b=24, l=16, r=16),
        )
        return fig
    except Exception:
        return None

# ─────────────────────────────────────────────
#  OVERVIEW METRICS
# ─────────────────────────────────────────────
def render_overview(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns.tolist()

    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric cols", len(num_cols))
    c4.metric("Missing values", int(df.isnull().sum().sum()))

    tab1, tab2, tab3 = st.tabs(["📄 Preview", "📊 Statistics", "🔍 Column Info"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True, height=280)

    with tab2:
        if num_cols:
            st.dataframe(df[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab3:
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Non-null": df.notnull().sum().values,
            "Null%": (df.isnull().mean() * 100).round(1).values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, height=280)

    if num_cols:
        st.markdown("#### Distribution of numeric columns")
        cols_to_show = num_cols[:6]
        ncols = min(3, len(cols_to_show))
        rows = [cols_to_show[i:i+ncols] for i in range(0, len(cols_to_show), ncols)]
        for row in rows:
            grid = st.columns(len(row))
            for col_widget, col_name in zip(grid, row):
                fig = px.histogram(
                    df, x=col_name, template=PLOTLY_TEMPLATE,
                    title=col_name, nbins=30,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig.update_layout(
                    plot_bgcolor="#141821", paper_bgcolor="#141821",
                    font=dict(family="IBM Plex Sans", color="#e2e8f0"),
                    margin=dict(t=40, b=20, l=8, r=8), showlegend=False,
                )
                col_widget.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗄️ SQL Analytics")
    st.markdown("---")

    # ── API Key input ──────────────────────────
    st.markdown("### 🔑 Groq API Key")
    api_key_input = st.text_input(
        "Enter your Groq API key",
        type="password",
        value=st.session_state.groq_api_key,
        placeholder="gsk_••••••••••••••••••••••",
        label_visibility="collapsed",
    )

    # Persist key in session state
    if api_key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key_input

    if st.session_state.groq_api_key:
        st.markdown("<p class='api-badge-ok'>✅ API key set</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='api-badge-err'>⚠️ No API key – queries disabled</p>", unsafe_allow_html=True)

    st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")

    # ── CSV upload ─────────────────────────────
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_new = pd.read_csv(uploaded)
        tbl = re.sub(r"[^a-zA-Z0-9_]", "_", uploaded.name.replace(".csv", "").lower())
        st.session_state.df = df_new
        st.session_state.table_name = tbl
        load_df_to_sqlite(df_new, tbl)
        st.success(f"Loaded `{tbl}` ({len(df_new):,} rows)")

    st.markdown("---")
    st.markdown("### 💬 Query History")

    if not st.session_state.query_history:
        st.caption("No queries yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history[-8:])):
            st.markdown(
                f"<div class='history-item'>#{len(st.session_state.query_history)-i}  {item['question'][:55]}…</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.session_state.last_result_df = None
        st.session_state.last_sql = ""
        st.session_state.last_fig = None
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("# SQL Analytics Dashboard")

if st.session_state.df is None:
    st.info("⬅️  Upload a CSV file from the sidebar to get started.")
    st.stop()

df = st.session_state.df
table_name = st.session_state.table_name

render_overview(df)

st.markdown("---")
st.markdown("### 🤖 Ask a Question")

col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "Natural language question",
        placeholder=f'e.g. "Show top 10 {df.columns[0]} by {df.columns[-1]}"',
        label_visibility="collapsed",
    )
with col_btn:
    ask = st.button("▶  Run", use_container_width=True)

num_cols = df.select_dtypes(include="number").columns.tolist()
st.caption(
    "**Quick starts:** "
    + "  `How many rows are in the dataset?`"
    + (f"  |  `What is the average {num_cols[0]}?`" if num_cols else "")
    + "  |  `Show the first 10 rows`"
    + f"  |  `Which {df.columns[0]} appears most frequently?`"
)

if ask and question.strip():
    # Guard: require API key before running
    if not st.session_state.groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar before running a query.")
        st.stop()

    schema = get_schema(df, table_name)
    active_key = st.session_state.groq_api_key

    with st.spinner("🧠 Generating SQL…"):
        try:
            sql = generate_sql(question, schema, active_key)
        except Exception as e:
            st.error(f"LLM Error: {e}")
            st.stop()
    st.session_state.last_sql = sql

    with st.spinner("⚙️ Running query…"):
        try:
            result_df = run_sql(sql)
            st.session_state.last_result_df = result_df
        except Exception as e:
            st.error(f"SQL Error: {e}")
            st.code(sql, language="sql")
            st.stop()

    with st.spinner("📊 Picking best visualization…"):
        viz_spec = generate_viz_spec(question, result_df, active_key)
        fig = build_chart(viz_spec, result_df)
        st.session_state.last_fig = fig
        st.session_state.last_viz_spec = viz_spec

    st.session_state.query_history.append({
        "question": question,
        "sql": sql,
        "rows": len(result_df),
    })

# ─────────────────────────────────────────────
#  RESULTS PANEL
# ─────────────────────────────────────────────
if st.session_state.last_result_df is not None:
    result_df = st.session_state.last_result_df
    sql      = st.session_state.last_sql
    fig      = st.session_state.last_fig
    spec     = st.session_state.get("last_viz_spec", {})

    st.markdown("---")

    r1, r2 = st.columns([2, 1])
    with r1:
        st.markdown("#### 📟 Generated SQL")
        st.code(sql, language="sql")
    with r2:
        st.markdown("#### 📈 Result Summary")
        st.metric("Rows returned", f"{len(result_df):,}")
        st.metric("Columns", len(result_df.columns))
        if spec.get("insight"):
            st.info(f"💡 {spec['insight']}")

    st.markdown("#### 📊 Visualization  &  Data")
    tab_chart, tab_data, tab_export = st.tabs(["Chart", "Table", "Export SQL"])

    with tab_chart:
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart generated for this query — showing table only.")
            st.dataframe(result_df, use_container_width=True)

    with tab_data:
        st.dataframe(result_df, use_container_width=True, height=400)

    with tab_export:
        st.markdown("Copy or download the raw SQL below:")
        st.code(sql, language="sql")
        st.download_button(
            "⬇️ Download SQL",
            data=sql,
            file_name="query.sql",
            mime="text/plain",
        )
        if len(result_df) > 0:
            csv_bytes = result_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
