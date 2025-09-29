import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ---- Configuration ----
APP_DIR = Path(os.getenv("CSAT_APP_DIR", "/content/csat_analysis_cache"))
APP_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_RESULTS_FILE = APP_DIR / "ai_csat_analysis_results.csv"
INTERPRETER_STATS_FILE = APP_DIR / "interpreter_performance_stats.csv"

# Rate-limiting / retry defaults
DEFAULT_SLEEP_BETWEEN_REQUESTS = float(os.getenv("CSAT_SLEEP", "0.2"))
DEFAULT_MAX_RETRIES = int(os.getenv("CSAT_MAX_RETRIES", "3"))

# Plot colors (single source of truth)
COLORS = {
    "Issues": "#ff6b6b",
    "Positive": "#51cf66",
    "Other": "#ffd43b",
}

# Configure logging (useful during development)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csat_analyzer")

# ---- Category Mappings ----
CATEGORY_DISPLAY_NAMES = {
    'audio_quality_poor': 'Poor Audio Quality',
    'audio_volume_low': 'Low Audio Volume',
    'audio_background_noise': 'Background Noise',
    'video_quality_poor': 'Poor Video Quality',
    'connection_problems': 'Connection Problems',
    'call_disconnected': 'Call Disconnected',
    'technical_difficulties': 'Technical Difficulties',
    'rude_behavior': 'Rude Behavior',
    'impatient_behavior': 'Impatient Behavior',
    'inappropriate_conduct': 'Inappropriate Conduct',
    'rushing_interpretation': 'Rushing/Hurried',
    'translation_errors': 'Translation Errors',
    'incomplete_interpretation': 'Incomplete Interpretation',
    'slow_interpretation': 'Slow Interpretation',
    'interpreter_absent': 'Interpreter Absent',
    'interpreter_unresponsive': 'Interpreter Unresponsive',
    'excellent_service': 'Excellent Service',
    'professional_courteous': 'Professional & Courteous',
    'accurate_translation': 'Accurate Translation',
    'patient_helpful': 'Patient & Helpful',
    'clear_communication': 'Clear Communication',
    'brief_positive': 'Brief Positive',
    'unclear_feedback': 'Unclear Feedback',
    'other': 'Other',
    'error': 'Error'
}

# Set of categories grouped by theme
ISSUE_CATEGORIES = {
    'audio_quality_poor', 'audio_volume_low', 'audio_background_noise', 'video_quality_poor',
    'connection_problems', 'call_disconnected', 'technical_difficulties',
    'rude_behavior', 'impatient_behavior', 'inappropriate_conduct', 'rushing_interpretation',
    'translation_errors', 'incomplete_interpretation', 'slow_interpretation',
    'interpreter_absent', 'interpreter_unresponsive'
}
POSITIVE_CATEGORIES = {
    'excellent_service', 'professional_courteous', 'accurate_translation',
    'patient_helpful', 'clear_communication', 'brief_positive'
}

# Severity weights (vectorizable)
SEVERITY_WEIGHTS = {
    'rude_behavior': 5,
    'inappropriate_conduct': 5,
    'translation_errors': 4,
    'interpreter_absent': 4,
    'interpreter_unresponsive': 4,
    'impatient_behavior': 3,
    'rushing_interpretation': 3,
    'incomplete_interpretation': 3,
    'slow_interpretation': 2,
    'audio_quality_poor': 2,
    'video_quality_poor': 2,
    'call_disconnected': 2,
    'connection_problems': 1,
    'audio_volume_low': 1,
    'audio_background_noise': 1,
    'technical_difficulties': 1
}

# ---- Ollama client ----
class OllamaFeedbackAnalyzer:
    """Lightweight client for Ollama endpoints with retries and simple failover."""

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434",
                 sleep_between_requests: float = DEFAULT_SLEEP_BETWEEN_REQUESTS,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_chat = f"{self.base_url}/api/chat"
        self.api_generate = f"{self.base_url}/api/generate"
        self.api_tags = f"{self.base_url}/api/tags"
        self.sleep_between_requests = sleep_between_requests
        self.max_retries = max_retries
        self.use_chat_endpoint = True
        self.available_models = []
        # Attempt to detect connection but do not raise; UI will surface issues
        try:
            self.available_models = self.test_connection()
        except Exception as e:
            logger.warning("Ollama connection check failed: %s", e)

    def _post_with_retries(self, url: str, payload: dict, timeout: int = 60) -> Optional[requests.Response]:
        """Post JSON to `url` with retries and backoff. Returns Response or None."""
        backoff = 1
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=timeout, headers={'Content-Type': 'application/json'})
                if resp.status_code == 200:
                    return resp
                else:
                    # log and attempt a retry for server errors
                    logger.warning("Request to %s returned status %s (attempt %s/%s).", url, resp.status_code, attempt, self.max_retries)
            except requests.exceptions.RequestException as exc:
                logger.warning("Request exception on attempt %s/%s: %s", attempt, self.max_retries, exc)
            time.sleep(backoff)
            backoff *= 1.5
        return None
    def analyze_feedback_batch(self, feedback_list: List[str], progress_callback=None) -> List[str]:
        """Analyze a list of feedback items in parallel with progress tracking and save results to CSV."""
        results = [None] * len(feedback_list)
        total = len(feedback_list)
        completed = 0

        max_workers = (os.cpu_count() or 8)*2  # safe default

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.categorize_single_feedback, fb): i
                for i, fb in enumerate(feedback_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(f"Error analyzing feedback index {idx}: {e}")
                    results[idx] = "error"

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        return results
    def test_connection(self) -> List[str]:
        """Check Ollama /models (tags) endpoint for available models."""
        try:
            resp = requests.get(self.api_tags, timeout=10)
            if resp.ok:
                data = resp.json()
                models = [m.get('name', '') for m in data.get('models', [])]
                # Normalize base names
                base_names = [m.split(':')[0] for m in models if m]
                st.sidebar.success("✅ Ollama reachable")
                st.sidebar.info(f"Models: {', '.join(base_names)}")
                # If specified model not present, pick first available
                base_model = self.model_name.split(':')[0]
                if base_model not in base_names and models:
                    st.sidebar.warning(f"⚠️ Model {self.model_name} not found locally. Will try {base_names[0]}")
                    self.model_name = base_names[0]
                return base_names
            else:
                st.sidebar.error(f"❌ Ollama returned status {resp.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            st.sidebar.error("❌ Cannot reach Ollama API.")
            logger.exception(e)
            return []

    def test_model_simple(self) -> bool:
        """Run a tiny prompt to validate the model endpoint (chat preferred, fallback to generate)."""
        test_prompt = "Say 'Hello' in one word."
        # Try chat endpoint
        chat_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": test_prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8}
        }
        resp = self._post_with_retries(self.api_chat, chat_payload, timeout=20)
        if resp:
            try:
                rj = resp.json()
                content = rj.get('message', {}).get('content', '').strip()
                st.success(f"✅ Chat test OK: {content}")
                self.use_chat_endpoint = True
                return True
            except Exception:
                pass

        # Fallback to generate
        gen_payload = {
            "model": self.model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 8}
        }
        resp = self._post_with_retries(self.api_generate, gen_payload, timeout=20)
        if resp:
            try:
                rj = resp.json()
                content = rj.get('response', '').strip()
                st.success(f"✅ Generate test OK: {content}")
                self.use_chat_endpoint = False
                return True
            except Exception:
                pass

        st.error("❌ Model test failed (both endpoints).")
        return False

    def query_ollama(self, prompt: str) -> str:
        """Query the selected endpoint and return text response or 'error'."""
        if self.use_chat_endpoint:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200, "top_p": 0.9}
            }
            endpoint = self.api_chat
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 200, "top_p": 0.9}
            }
            endpoint = self.api_generate

        resp = self._post_with_retries(endpoint, payload, timeout=60)
        if not resp:
            return "error"

        try:
            rj = resp.json()
            if self.use_chat_endpoint:
                return rj.get('message', {}).get('content', '').strip()
            return rj.get('response', '').strip()
        except Exception as e:
            logger.exception("Failed to parse Ollama JSON response: %s", e)
            return "error"

    def categorize_single_feedback(self, feedback: str) -> str:
        """Categorize a single feedback string using the LLM, with simple fallback mapping."""
        if not feedback or pd.isna(feedback) or not feedback.strip():
            return "other"

        prompt = create_categorization_prompt(feedback)
        result = self.query_ollama(prompt)
        if result == "error":
            return "error"

        result_clean = result.strip().lower()
        # If the model returned one of the canonical categories, return it
        if result_clean in CATEGORY_DISPLAY_NAMES:
            return result_clean

        # Try to match known keywords
        keyword_map = {
            'audio': 'audio_quality_poor',
            'volume': 'audio_volume_low',
            'background': 'audio_background_noise',
            'video': 'video_quality_poor',
            'disconnect': 'call_disconnected',
            'connection': 'connection_problems',
            'rude': 'rude_behavior',
            'professional': 'professional_courteous',
            'excellent': 'excellent_service',
            'translation': 'translation_errors',
            'patient': 'patient_helpful'
        }
        for k, v in keyword_map.items():
            if k in result_clean:
                return v

        return 'other'

# ---- Prompt builder ----
def create_categorization_prompt(feedback: str) -> str:
    """Return the prompt for the LLM for categorization (kept concise)."""
    categories_text = "\n".join([f"- {k}: {CATEGORY_DISPLAY_NAMES.get(k, k)}" for k in CATEGORY_DISPLAY_NAMES.keys() if k != 'error'])
    prompt = (
        "You are an assistant that maps a short customer feedback about a medical interpreter service "
        "to exactly one category. Respond with only the category key.\n\n"
        f"Categories:\n{categories_text}\n\n"
        f"Feedback: \"{feedback}\"\n\nRespond with ONLY the category key (e.g. rude_behavior)."
    )
    return prompt

# ---- File utilities (cached where appropriate) ----
@st.cache_data(ttl=600)
def load_existing_analysis() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load saved analysis and interpreter stats if files exist and have expected columns."""
    if ANALYSIS_RESULTS_FILE.exists():
        try:
            df = pd.read_csv(ANALYSIS_RESULTS_FILE)
            required_cols = {'ClientFeedback', 'specific_category', 'theme', 'interpreter_info'}
            if required_cols.issubset(df.columns):
                interpreter_stats = None
                if INTERPRETER_STATS_FILE.exists():
                    try:
                        interpreter_stats = pd.read_csv(INTERPRETER_STATS_FILE)
                    except Exception:
                        interpreter_stats = None
                return df, interpreter_stats
            else:
                st.warning("Existing analysis missing required columns. Upload new data.")
        except Exception as e:
            st.error(f"Could not load saved analysis: {e}")
    return None, None

def save_analysis_results(df_analyzed: pd.DataFrame, interpreter_stats: Optional[pd.DataFrame]) -> None:
    """Save analysis and interpreter stats safely."""
    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        df = df_analyzed.copy()
        df['analysis_timestamp'] = ts
        df.to_csv(ANALYSIS_RESULTS_FILE, index=False)
        if interpreter_stats is not None:
            stats = interpreter_stats.copy()
            stats['analysis_timestamp'] = ts
            stats.to_csv(INTERPRETER_STATS_FILE, index=False)
        st.success(f"✅ Saved analysis to {ANALYSIS_RESULTS_FILE}")
    except Exception as e:
        st.error(f"❌ Failed to save results: {e}")
        logger.exception(e)

# ---- Data loading & cleaning ----
def load_and_clean_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load CSV and create helpful columns used across the app."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

    if 'ClientFeedback' not in df.columns:
        st.error("CSV must contain 'ClientFeedback' column.")
        return None

    df = df.copy()
    df['feedback_clean'] = df['ClientFeedback'].fillna('').astype(str).str.strip()
    df = df[df['feedback_clean'] != ""].reset_index(drop=True)

    # Safe creation of interpreter fields (if missing columns, fill with blanks)
    first = df.get('InterpreterFirstName', pd.Series([""] * len(df))).fillna("")
    last = df.get('InterpreterLastName', pd.Series([""] * len(df))).fillna("")
    interp_id = df.get('InterpreterId', pd.Series([""] * len(df))).fillna("")

    df['interpreter_name'] = (first.astype(str) + " " + last.astype(str)).str.strip()
    df['interpreter_info'] = (df['interpreter_name'] + " (ID: " + interp_id.astype(str) + ")").str.strip()
    # Keep InterpreterId column for grouping operations
    if 'InterpreterId' not in df.columns:
        df['InterpreterId'] = interp_id

    return df

# ---- Analysis & Visualization ----
def analyze_feedback_with_ollama(df: pd.DataFrame, analyzer: OllamaFeedbackAnalyzer) -> pd.DataFrame:
    """Run the LLM categorization and add 'specific_category' + 'theme' columns."""
    total = len(df)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress_callback(current, total_local):
        progress_bar.progress(min(1.0, current / total_local))
        progress_text.text(f"AI analyzing feedback {current}/{total_local}")

    feedbacks = df['feedback_clean'].tolist()
    categories = analyzer.analyze_feedback_batch(feedbacks, progress_callback)
    progress_bar.empty()
    progress_text.empty()

    df = df.copy()
    df['specific_category'] = categories
    df['theme'] = df['specific_category'].apply(lambda x: 'Issues' if x in ISSUE_CATEGORIES
                                               else 'Positive' if x in POSITIVE_CATEGORIES
                                               else 'Other')
    err_count = sum(1 for c in categories if c == 'error')
    if err_count:
        st.warning(f"⚠️ {err_count} items could not be analyzed due to API errors")
    return df

def analyze_interpreter_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate interpreter performance and compute severity scores vectorized."""
    df_clean = df[df['specific_category'] != 'error'].copy()
    if df_clean.empty:
        return pd.DataFrame(columns=['InterpreterId', 'interpreter_name', 'interpreter_info', 'total_feedback',
                                     'issue_count', 'positive_count', 'issue_percentage', 'severity_score'])

    # Basic counts
    group = df_clean.groupby(['InterpreterId', 'interpreter_name', 'interpreter_info'], as_index=False).agg(
        total_feedback=('specific_category', 'count'),
        issue_count=('theme', lambda x: (x == 'Issues').sum())
    )
    group['positive_count'] = group['total_feedback'] - group['issue_count']
    group['issue_percentage'] = (group['issue_count'] / group['total_feedback'] * 100).round(1)

    # Weighted severity: join category counts then multiply by weights
    cat_counts = df_clean.groupby(['InterpreterId', 'specific_category']).size().unstack(fill_value=0)
    # Ensure all keys from SEVERITY_WEIGHTS exist
    for k in SEVERITY_WEIGHTS.keys():
        if k not in cat_counts.columns:
            cat_counts[k] = 0
    # compute severity via dot product
    weight_series = pd.Series(SEVERITY_WEIGHTS)
    # Align columns and multiply (only the ones present)
    severity = cat_counts[list(weight_series.index)].dot(weight_series).rename('severity_score')
    severity = severity.reset_index()
    # Merge with group
    result = group.merge(severity, on='InterpreterId', how='left').fillna({'severity_score': 0})
    result = result.sort_values(['severity_score', 'issue_percentage'], ascending=False).reset_index(drop=True)
    return result

def create_visualizations(df: pd.DataFrame) -> Tuple[px.pie, px.pie, Optional[px.bar]]:
    """Return (overview_pie, specific_pie, issues_bar or None)."""
    df_clean = df[df['specific_category'] != 'error']
    theme_counts = df_clean['theme'].value_counts()
    # Main overview pie
    fig_pie_main = px.pie(values=theme_counts.values, names=theme_counts.index,
                          title="AI-Powered Feedback Analysis - Overall Results",
                          color_discrete_map=COLORS)

    # Specific categories pie
    category_counts = df_clean['specific_category'].value_counts()
    display_names = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in category_counts.index]
    # color sequence based on category theme
    def seq_for(cat_index):
        seq = []
        for c in cat_index:
            if c in ISSUE_CATEGORIES:
                seq.append(COLORS['Issues'])
            elif c in POSITIVE_CATEGORIES:
                seq.append(COLORS['Positive'])
            else:
                seq.append(COLORS['Other'])
        return seq

    fig_pie_specific = px.pie(values=category_counts.values, names=display_names,
                              title="Specific Issues & Feedback Categories (AI-Detected)",
                              color_discrete_sequence=seq_for(category_counts.index))

    # Issues bar chart (top 10)
    issue_df = df_clean[df_clean['theme'] == 'Issues']
    if not issue_df.empty:
        issue_counts = issue_df['specific_category'].value_counts().head(10)
        issue_display = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in issue_counts.index]
        fig_bar = px.bar(x=issue_counts.values, y=issue_display, orientation='h',
                         title="Top 10 Issues Identified by AI Analysis",
                         labels={'x': 'Number of Cases', 'y': 'Issue Type'},
                         color=issue_counts.values)
        fig_bar.update_layout(showlegend=False, height=400)
    else:
        fig_bar = None

    return fig_pie_main, fig_pie_specific, fig_bar

# ---- UI / Streamlit app ----
def display_detailed_results(df: pd.DataFrame, interpreter_stats: pd.DataFrame):
    """UI for showing metrics, tables and examples."""
    df_clean = df[df['specific_category'] != 'error']
    total_feedback = len(df)
    total_analyzed = len(df_clean)
    issues_count = (df_clean['theme'] == 'Issues').sum()
    positive_count = (df_clean['theme'] == 'Positive').sum()
    other_count = (df_clean['theme'] == 'Other').sum()
    error_count = (df['specific_category'] == 'error').sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Feedback", total_feedback)
    c2.metric("Successfully Analyzed", total_analyzed)
    c3.metric("Issues Identified", f"{issues_count} ({(issues_count/total_analyzed*100) if total_analyzed else 0:.0f}%)")
    c4.metric("Positive Feedback", f"{positive_count} ({(positive_count/total_analyzed*100) if total_analyzed else 0:.0f}%)")
    if error_count:
        c5.metric("Analysis Errors", error_count, delta="⚠️")

    st.subheader("🚨 Interpreter Performance Analysis")
    problem_interpreters = interpreter_stats[
        (interpreter_stats['issue_count'] > 0) &
        ((interpreter_stats['issue_percentage'] > 50) | (interpreter_stats['severity_score'] > 5))
    ].head(10)

    if not problem_interpreters.empty:
        st.warning(f"⚠️ Found {len(problem_interpreters)} interpreters with concerning feedback patterns:")
        display_cols = ['interpreter_info', 'total_feedback', 'issue_count', 'issue_percentage', 'severity_score']
        table = problem_interpreters[display_cols].copy()
        table.columns = ['Interpreter (Name & ID)', 'Total Feedback', 'Issues', 'Issue %', 'Severity Score']
        st.dataframe(table.style.format({'Issue %': '{:.1f}'}), use_container_width=True)

        st.subheader("🔍 Issues by Problem Interpreters")
        for _, interpreter in problem_interpreters.head(5).iterrows():
            with st.expander(f"Issues for {interpreter['interpreter_name']} (ID: {interpreter['InterpreterId']}) - {int(interpreter['issue_count'])} issues"):
                interpreter_issues = df_clean[
                    (df_clean['InterpreterId'] == interpreter['InterpreterId']) &
                    (df_clean['theme'] == 'Issues')
                ]
                for _, row in interpreter_issues.iterrows():
                    display = CATEGORY_DISPLAY_NAMES.get(row['specific_category'], row['specific_category'])
                    st.write(f"**{display}**: \"{row['feedback_clean']}\"")
    else:
        st.success("✅ No major interpreter performance concerns identified!")

    st.subheader("📊 All Interpreter Performance Summary")
    all_interpreters_display = interpreter_stats[interpreter_stats['total_feedback'] > 0][[
        'interpreter_info', 'total_feedback', 'issue_count', 'positive_count', 'issue_percentage'
    ]].copy()
    all_interpreters_display.columns = ['Interpreter (Name & ID)', 'Total Feedback', 'Issues', 'Positive', 'Issue %']
    st.dataframe(all_interpreters_display, use_container_width=True)

    st.subheader("🔴 Detailed Issues Breakdown")
    for cat, label in CATEGORY_DISPLAY_NAMES.items():
        if cat in ISSUE_CATEGORIES:
            cat_df = df_clean[df_clean['specific_category'] == cat]
            if not cat_df.empty:
                with st.expander(f"{label} ({len(cat_df)} cases)"):
                    for _, r in cat_df.iterrows():
                        st.write(f"**{r['interpreter_info']}**: \"{r['feedback_clean']}\"")

    st.subheader("🟢 Positive Feedback Examples")
    for cat in POSITIVE_CATEGORIES:
        label = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        cat_df = df_clean[df_clean['specific_category'] == cat]
        if not cat_df.empty:
            with st.expander(f"{label} ({len(cat_df)} cases)"):
                for _, r in cat_df.head(5).iterrows():
                    st.write(f"**{r['interpreter_info']}**: \"{r['feedback_clean']}\"")

def main():
    st.set_page_config(page_title="AI-Powered CSAT Feedback Analyzer", page_icon="🤖", layout="wide")
    st.title("🤖 AI-Powered CSAT Feedback Analysis")
    st.markdown("Enhanced Medical Interpreter Service Feedback Analyzer")

    existing_df, existing_stats = load_existing_analysis()
    if existing_df is not None:
        if 'analysis_timestamp' in existing_df.columns:
            st.success(f"📊 Loaded Previous Analysis (Generated: {existing_df['analysis_timestamp'].iloc[0]})")
        else:
            st.success("📊 Loaded Previous Analysis")
        st.info(f"Found {len(existing_df)} analyzed feedback entries from {existing_df['interpreter_name'].nunique()} interpreters")

        st.subheader("📈 Analysis Results (From Saved Data)")
        fig_pie_main, fig_pie_specific, fig_bar = create_visualizations(existing_df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie_main, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie_specific, use_container_width=True)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        display_detailed_results(existing_df, existing_stats if existing_stats is not None else analyze_interpreter_performance(existing_df))
        st.divider()

    st.subheader("📤 Upload New Data or Update Analysis")
    st.info("Upload a CSV file with at minimum: ClientFeedback, InterpreterFirstName, InterpreterLastName, InterpreterId")

    # Sidebar: Model & Ollama settings
    st.sidebar.header("🤖 AI Model Settings")
    model_options = ["llama3.2", "llama3.2:1b", "mistral", "phi3", "gemma2", "qwen2.5"]
    selected_model = st.sidebar.selectbox("Select Ollama Model", model_options, index=0)
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = OllamaFeedbackAnalyzer(model_name=selected_model, base_url=ollama_url)
        except Exception as e:
            logger.exception(e)
            st.session_state.analyzer = OllamaFeedbackAnalyzer(model_name=selected_model, base_url=ollama_url)

    # Option to reinitialize / test
    if st.sidebar.button("🧪 Test Model"):
        st.session_state.analyzer = OllamaFeedbackAnalyzer(model_name=selected_model, base_url=ollama_url)
        st.session_state.analyzer.test_model_simple()

    uploaded_file = st.file_uploader("Upload CSAT CSV file", type=['csv'])

    if uploaded_file is not None:
        with st.spinner("Loading and cleaning data..."):
            df = load_and_clean_data(uploaded_file)

        if df is None:
            return

        st.success(f"✅ Loaded {len(df)} feedback entries from {df['interpreter_name'].nunique()} interpreters")
        st.subheader("📋 Sample Feedback")
        for _, row in df[['interpreter_info', 'feedback_clean']].head(3).iterrows():
            st.write(f"**{row['interpreter_info']}**: \"{row['feedback_clean']}\"")

        if not st.session_state.analyzer.available_models:
            st.error("❌ Ollama doesn't appear accessible or no models found. Check the URL and that Ollama is running.")
            st.stop()

        if st.button("🚀 Start Enhanced AI Analysis"):
            if not st.session_state.analyzer.test_model_simple():
                st.error("❌ Model test failed - cannot proceed.")
                st.stop()

            with st.spinner("🤖 AI analyzing..."):
                df_analyzed = analyze_feedback_with_ollama(df, st.session_state.analyzer)

            st.success(f"✅ Enhanced AI analysis complete! Processed {len(df_analyzed)} entries")
            with st.spinner("👤 Analyzing interpreter performance..."):
                interpreter_stats = analyze_interpreter_performance(df_analyzed)
            with st.spinner("💾 Saving analysis results..."):
                save_analysis_results(df_analyzed, interpreter_stats)

            st.info("🧠 Enhanced AI Analysis Active: specific issue categorization and severity scoring")
            st.subheader("📈 Results")
            fig_pie_main, fig_pie_specific, fig_bar = create_visualizations(df_analyzed)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie_main, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pie_specific, use_container_width=True)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
            display_detailed_results(df_analyzed, interpreter_stats)

            # Prepare safe download (pick columns that exist)
            possible_cols = [
                'Months in est_time_Zone', 'InterpreterCompanyName', 'InterpreterId',
                'InterpreterFirstName', 'InterpreterLastName', 'TargetLanguage',
                'ClientFeedback', 'interpreter_info', 'specific_category', 'theme'
            ]
            download_cols = [c for c in possible_cols if c in df_analyzed.columns]
            download_df = df_analyzed[download_cols].copy()
            csv_bytes = download_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Enhanced AI Analysis Results", data=csv_bytes,
                               file_name="enhanced_ai_csat_analysis_results.csv", mime="text/csv")


if __name__ == "__main__":
    main()
