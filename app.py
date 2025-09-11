import streamlit as st
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go
import requests
import time
import os
from typing import List
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="AI-Powered CSAT Feedback Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Constants
ANALYSIS_RESULTS_FILE = "ai_csat_analysis_results.csv"
INTERPRETER_STATS_FILE = "interpreter_performance_stats.csv"

class OllamaFeedbackAnalyzer:
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url_chat = f"{base_url}/api/chat"
        self.api_url_generate = f"{base_url}/api/generate"
        self.api_url_tags = f"{base_url}/api/tags"

        # Test connection and setup
        self.available_models = self.test_connection()
        self.use_chat_endpoint = True  # Prefer chat endpoint

    def test_connection(self):
        """Test if Ollama is running and get available models"""
        try:
            # Test basic connectivity
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)

            if response.status_code == 200:
                models_data = response.json()
                available_models = [m['name'].split(':')[0] for m in models_data.get('models', [])]

                st.sidebar.success(f"✅ Ollama connected successfully!")
                st.sidebar.info(f"📦 Available models: {', '.join(available_models)}")

                # Check if selected model is available
                model_base_name = self.model_name.split(':')[0]
                if model_base_name not in available_models and self.model_name not in [m['name'] for m in models_data.get('models', [])]:
                    st.sidebar.warning(f"⚠️ Model '{self.model_name}' not found!")
                    st.sidebar.info(f"💡 Try: `ollama pull {self.model_name}`")
                    if available_models:
                        suggested_model = available_models[0]
                        st.sidebar.info(f"🔄 Will try to use: {suggested_model}")
                        self.model_name = suggested_model

                return available_models

            else:
                st.sidebar.error(f"❌ Ollama API responded with status {response.status_code}")
                return []

        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Cannot connect to Ollama!")
            st.sidebar.error("🔧 **Fix Steps:**")
            st.sidebar.error("1. Start Ollama: `ollama serve`")
            st.sidebar.error("2. Pull a model: `ollama pull llama3.2`")
            st.sidebar.error("3. Check: http://localhost:11434")
            return []
        except Exception as e:
            st.sidebar.error(f"❌ Unexpected error: {e}")
            return []

    def test_model_simple(self):
        """Test if the model works with a simple prompt"""
        test_prompt = "Say 'Hello' in one word."

        try:
            st.info(f"🧪 Testing model '{self.model_name}' with simple prompt...")

            # Try chat endpoint first
            chat_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10  # Very short response
                }
            }

            response = requests.post(
                self.api_url_chat,
                json=chat_payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                test_response = result.get('message', {}).get('content', '').strip()
                st.success(f"✅ Model test successful! Response: '{test_response}'")
                return True
            else:
                st.error(f"❌ Chat endpoint failed (status {response.status_code})")
                st.error(f"Response: {response.text}")

                # Try generate endpoint as fallback
                st.info("🔄 Trying generate endpoint...")
                generate_payload = {
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                }

                response = requests.post(
                    self.api_url_generate,
                    json=generate_payload,
                    timeout=30,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    result = response.json()
                    test_response = result.get('response', '').strip()
                    st.success(f"✅ Generate endpoint works! Response: '{test_response}'")
                    self.use_chat_endpoint = False
                    return True
                else:
                    st.error(f"❌ Both endpoints failed!")
                    st.error(f"Generate response: {response.text}")
                    return False

        except requests.exceptions.Timeout:
            st.error("❌ Model test timed out! Model might be too large or slow.")
            return False
        except Exception as e:
            st.error(f"❌ Model test failed: {e}")
            return False

    def query_ollama(self, prompt: str, max_retries: int = 2) -> str:
        """Send a query to Ollama with proper error handling"""

        if self.use_chat_endpoint:
            # Use chat endpoint (recommended)
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                    "top_p": 0.9
                }
            }
            endpoint = self.api_url_chat
        else:
            # Use generate endpoint (fallback)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                    "top_p": 0.9
                }
            }
            endpoint = self.api_url_generate

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=60,  # Increased timeout
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    result = response.json()

                    if self.use_chat_endpoint:
                        return result.get('message', {}).get('content', '').strip()
                    else:
                        return result.get('response', '').strip()
                else:
                    error_msg = f"API error {response.status_code}: {response.text[:200]}"
                    if attempt == max_retries - 1:
                        st.error(f"❌ {error_msg}")
                    return "error"

            except requests.exceptions.Timeout:
                error_msg = f"Request timeout on attempt {attempt + 1}"
                if attempt == max_retries - 1:
                    st.error(f"❌ {error_msg}")
                return "error"
            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: {e}"
                if attempt == max_retries - 1:
                    st.error(f"❌ {error_msg}")
                return "error"

            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2)

        return "error"

    def create_categorization_prompt(self, feedback: str) -> str:
        """Create a structured prompt for categorizing feedback into specific categories"""

        prompt = f"""Analyze this customer feedback for a medical interpreter service. Categorize it into ONE of these SPECIFIC categories:

TECHNICAL ISSUES:
- audio_quality_poor: Poor audio quality, unclear sound, distorted audio
- audio_volume_low: Volume too low, can't hear clearly
- audio_background_noise: Background noise, echo, interference
- video_quality_poor: Poor video quality, blurry, frozen video
- connection_problems: Problems connecting to the service, login issues
- call_disconnected: Calls being disconnected, dropped calls, hung up
- technical_difficulties: General technical problems, platform issues

INTERPRETER BEHAVIOR ISSUES:
- rude_behavior: Rude, disrespectful, unprofessional behavior
- impatient_behavior: Impatient, frustrated, dismissive attitude
- inappropriate_conduct: Inappropriate comments or behavior
- rushing_interpretation: Rushing, being in a hurry, too fast

SERVICE QUALITY ISSUES:
- translation_errors: Incorrect translation, wrong interpretation, accuracy problems
- incomplete_interpretation: Not interpreting everything, missing parts
- slow_interpretation: Too slow, long delays in interpretation
- interpreter_absent: Interpreter not available, not present, no show
- interpreter_unresponsive: Interpreter not responding, silent, unengaged

POSITIVE FEEDBACK:
- excellent_service: Excellent, exceptional, amazing, fantastic service
- professional_courteous: Professional, courteous, respectful, polite behavior
- accurate_translation: Accurate, correct, precise translation/interpretation
- patient_helpful: Patient, helpful, understanding, supportive
- clear_communication: Clear, easy to understand, good communication
- brief_positive: Brief positive responses like "good", "great", "thank you"

OTHER:
- unclear_feedback: Unclear, confusing, or mixed feedback
- other: Doesn't fit any above categories

CRITICAL: Focus on the MAIN issue. If feedback mentions multiple things, pick the PRIMARY concern.

Examples:
- "Audio was terrible" → audio_quality_poor
- "Interpreter was rude" → rude_behavior
- "Best translator ever" → excellent_service
- "Couldn't connect" → connection_problems
- "Too fast, rushing through" → rushing_interpretation

Feedback: "{feedback}"

Respond with ONLY the category name (e.g., "rude_behavior")."""

        return prompt

    def categorize_single_feedback(self, feedback: str) -> str:
        """Categorize a single piece of feedback using Ollama"""
        if not feedback or pd.isna(feedback) or feedback.strip() == "":
            return "other"

        prompt = self.create_categorization_prompt(feedback)
        result = self.query_ollama(prompt)

        # Define all valid specific categories
        valid_categories = {
            # Technical Issues
            'audio_quality_poor', 'audio_volume_low', 'audio_background_noise', 'video_quality_poor',
            'connection_problems', 'call_disconnected', 'technical_difficulties',
            # Behavior Issues
            'rude_behavior', 'impatient_behavior', 'inappropriate_conduct', 'rushing_interpretation',
            # Service Quality Issues
            'translation_errors', 'incomplete_interpretation', 'slow_interpretation',
            'interpreter_absent', 'interpreter_unresponsive',
            # Positive
            'excellent_service', 'professional_courteous', 'accurate_translation',
            'patient_helpful', 'clear_communication', 'brief_positive',
            # Other
            'unclear_feedback', 'other'
        }

        if result == "error":
            return "error"
        elif result.lower() in valid_categories:
            return result.lower()
        else:
            # If LLM returned something unexpected, try to map it intelligently
            result_lower = result.lower()

            # Keyword-based fallback mapping
            keyword_mapping = {
                'audio': 'audio_quality_poor',
                'video': 'video_quality_poor',
                'rude': 'rude_behavior',
                'professional': 'professional_courteous',
                'excellent': 'excellent_service',
                'connection': 'connection_problems',
                'disconnect': 'call_disconnected',
                'translation': 'translation_errors',
                'patient': 'patient_helpful'
            }

            for keyword, category in keyword_mapping.items():
                if keyword in result_lower:
                    return category

            return 'other'

    def analyze_feedback_batch(self, feedback_list: List[str], progress_callback=None) -> List[str]:
        """Analyze a batch of feedback with progress tracking"""
        results = []
        total = len(feedback_list)

        for i, feedback in enumerate(feedback_list):
            if progress_callback:
                progress_callback(i + 1, total)

            category = self.categorize_single_feedback(feedback)
            results.append(category)

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)

        return results

def load_existing_analysis():
    """Load existing analysis results if they exist"""
    if os.path.exists(ANALYSIS_RESULTS_FILE):
        try:
            df = pd.read_csv(ANALYSIS_RESULTS_FILE)

            # Check if required columns exist
            required_cols = ['ClientFeedback', 'specific_category', 'theme', 'interpreter_info']
            if all(col in df.columns for col in required_cols):

                # Load interpreter stats if available
                interpreter_stats = None
                if os.path.exists(INTERPRETER_STATS_FILE):
                    try:
                        interpreter_stats = pd.read_csv(INTERPRETER_STATS_FILE)
                    except:
                        pass

                return df, interpreter_stats
            else:
                st.warning("⚠️ Existing analysis file is missing required columns. Please upload new data.")
                return None, None

        except Exception as e:
            st.error(f"❌ Error loading existing analysis: {e}")
            return None, None

    return None, None

def save_analysis_results(df_analyzed, interpreter_stats):
    """Save analysis results to CSV files"""
    try:
        # Add timestamp to track when analysis was done
        df_analyzed['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save main analysis results
        df_analyzed.to_csv(ANALYSIS_RESULTS_FILE, index=False)

        # Save interpreter stats
        if interpreter_stats is not None:
            interpreter_stats['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            interpreter_stats.to_csv(INTERPRETER_STATS_FILE, index=False)

        st.success(f"✅ Analysis results saved to {ANALYSIS_RESULTS_FILE}")

    except Exception as e:
        st.error(f"❌ Error saving analysis results: {e}")

def load_and_clean_data(uploaded_file):
    """Load and clean the CSV data"""
    df = pd.read_csv(uploaded_file)

    # Clean the feedback column
    if 'ClientFeedback' in df.columns:
        df['feedback_clean'] = df['ClientFeedback'].fillna('').str.strip()
        df = df[df['feedback_clean'] != '']  # Remove empty feedback

        # Create interpreter full name and info
        df['interpreter_name'] = (df['InterpreterFirstName'].fillna('') + ' ' +
                                df['InterpreterLastName'].fillna('')).str.strip()
        df['interpreter_info'] = (df['interpreter_name'] + ' (ID: ' +
                                df['InterpreterId'].astype(str) + ')').str.strip()

        return df
    else:
        st.error("CSV must contain a 'ClientFeedback' column")
        return None

def analyze_feedback_with_ollama(df, analyzer):
    """Perform AI-powered analysis using Ollama"""

    # Create progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(current, total):
        progress = current / total
        progress_bar.progress(progress)
        progress_text.text(f"AI analyzing feedback {current}/{total}... ({progress:.0%})")

    # Analyze feedback using Ollama
    feedback_list = df['feedback_clean'].tolist()
    categories = analyzer.analyze_feedback_batch(feedback_list, update_progress)

    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()

    # Add results to dataframe
    df['specific_category'] = categories

    # Count errors
    error_count = sum(1 for cat in categories if cat == 'error')
    if error_count > 0:
        st.warning(f"⚠️ {error_count} items could not be analyzed due to API errors")

    # Create broader theme groupings for the specific categories
    issue_categories = [
        'audio_quality_poor', 'audio_volume_low', 'audio_background_noise', 'video_quality_poor',
        'connection_problems', 'call_disconnected', 'technical_difficulties',
        'rude_behavior', 'impatient_behavior', 'inappropriate_conduct', 'rushing_interpretation',
        'translation_errors', 'incomplete_interpretation', 'slow_interpretation',
        'interpreter_absent', 'interpreter_unresponsive'
    ]

    positive_categories = [
        'excellent_service', 'professional_courteous', 'accurate_translation',
        'patient_helpful', 'clear_communication', 'brief_positive'
    ]

    df['theme'] = df['specific_category'].apply(lambda x: 'Issues' if x in issue_categories
                                               else 'Positive' if x in positive_categories
                                               else 'Other')

    return df

def create_visualizations(df):
    """Create charts and visualizations"""

    # Filter out errors for visualization
    df_clean = df[df['specific_category'] != 'error']

    # Overall sentiment breakdown
    theme_counts = df_clean['theme'].value_counts()
    fig_pie_main = px.pie(
        values=theme_counts.values,
        names=theme_counts.index,
        title="AI-Powered Feedback Analysis - Overall Results",
        color_discrete_map={'Issues': '#ff6b6b', 'Positive': '#51cf66', 'Other': '#ffd43b'}
    )

    # Specific categories breakdown (more detailed)
    category_counts = df_clean['specific_category'].value_counts()

    # Create display names for categories
    category_display_names = {
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
        'other': 'Other'
    }

    # Get display names for the pie chart
    display_names = [category_display_names.get(cat, cat) for cat in category_counts.index]

    # Create color mapping for specific categories
    issue_cats = ['audio_quality_poor', 'audio_volume_low', 'audio_background_noise', 'video_quality_poor',
                  'connection_problems', 'call_disconnected', 'technical_difficulties',
                  'rude_behavior', 'impatient_behavior', 'inappropriate_conduct', 'rushing_interpretation',
                  'translation_errors', 'incomplete_interpretation', 'slow_interpretation',
                  'interpreter_absent', 'interpreter_unresponsive']

    positive_cats = ['excellent_service', 'professional_courteous', 'accurate_translation',
                     'patient_helpful', 'clear_communication', 'brief_positive']

    colors = []
    for cat in category_counts.index:
        if cat in issue_cats:
            colors.append('#ff6b6b')  # Red variants for issues
        elif cat in positive_cats:
            colors.append('#51cf66')  # Green variants for positive
        else:
            colors.append('#ffd43b')  # Yellow for other

    fig_pie_specific = px.pie(
        values=category_counts.values,
        names=display_names,
        title="Specific Issues & Feedback Categories (AI-Detected)",
        color_discrete_sequence=colors
    )

    # Top issues bar chart
    issue_df = df_clean[df_clean['theme'] == 'Issues']
    if not issue_df.empty:
        issue_counts = issue_df['specific_category'].value_counts().head(10)
        issue_display_names = [category_display_names.get(cat, cat) for cat in issue_counts.index]

        fig_bar = px.bar(
            x=issue_counts.values,
            y=issue_display_names,
            orientation='h',
            title="Top 10 Issues Identified by AI Analysis",
            labels={'x': 'Number of Cases', 'y': 'Issue Type'},
            color=issue_counts.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(showlegend=False, height=400)
    else:
        fig_bar = None

    return fig_pie_main, fig_pie_specific, fig_bar

def analyze_interpreter_performance(df):
    """Analyze interpreter performance and identify problem interpreters"""
    # Filter out errors
    df_clean = df[df['specific_category'] != 'error'].copy()

    # Group by interpreter
    interpreter_stats = df_clean.groupby(['InterpreterId', 'interpreter_name', 'interpreter_info']).agg({
        'specific_category': 'count',
        'theme': lambda x: (x == 'Issues').sum(),
    }).reset_index()

    interpreter_stats.columns = ['InterpreterId', 'interpreter_name', 'interpreter_info', 'total_feedback', 'issue_count']
    interpreter_stats['positive_count'] = interpreter_stats['total_feedback'] - interpreter_stats['issue_count']
    interpreter_stats['issue_percentage'] = (interpreter_stats['issue_count'] / interpreter_stats['total_feedback'] * 100).round(1)

    # Calculate severity score (weight different types of issues)
    severity_weights = {
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

    # Calculate weighted severity for each interpreter
    def calculate_severity(interpreter_id):
        interpreter_feedback = df_clean[df_clean['InterpreterId'] == interpreter_id]
        total_severity = 0
        for _, row in interpreter_feedback.iterrows():
            category = row['specific_category']
            weight = severity_weights.get(category, 0)
            total_severity += weight
        return total_severity

    interpreter_stats['severity_score'] = interpreter_stats['InterpreterId'].apply(calculate_severity)

    # Sort by severity and issue percentage
    interpreter_stats = interpreter_stats.sort_values(['severity_score', 'issue_percentage'], ascending=False)

    return interpreter_stats

def display_detailed_results(df, interpreter_stats):
    """Display detailed breakdown tables with AI insights"""

    # Filter out errors for analysis
    df_clean = df[df['specific_category'] != 'error']

    # Summary statistics
    total_feedback = len(df)
    total_analyzed = len(df_clean)
    issues_count = len(df_clean[df_clean['theme'] == 'Issues'])
    positive_count = len(df_clean[df_clean['theme'] == 'Positive'])
    other_count = len(df_clean[df_clean['theme'] == 'Other'])
    error_count = len(df[df['specific_category'] == 'error'])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("Successfully Analyzed", total_analyzed)
    with col3:
        st.metric("Issues Identified", f"{issues_count} ({issues_count/total_analyzed*100:.0f}%)" if total_analyzed > 0 else "0")
    with col4:
        st.metric("Positive Feedback", f"{positive_count} ({positive_count/total_analyzed*100:.0f}%)" if total_analyzed > 0 else "0")
    with col5:
        if error_count > 0:
            st.metric("Analysis Errors", error_count, delta="⚠️")

    # Problem Interpreters Section
    st.subheader("🚨 Interpreter Performance Analysis")

    # Show interpreters with most concerning feedback
    problem_interpreters = interpreter_stats[
        (interpreter_stats['issue_count'] > 0) &
        ((interpreter_stats['issue_percentage'] > 50) | (interpreter_stats['severity_score'] > 5))
    ].head(10)

    if not problem_interpreters.empty:
        st.warning(f"⚠️ Found {len(problem_interpreters)} interpreters with concerning feedback patterns:")

        # Create a more detailed table
        display_cols = ['interpreter_info', 'total_feedback', 'issue_count', 'issue_percentage', 'severity_score']
        problem_table = problem_interpreters[display_cols].copy()
        problem_table.columns = ['Interpreter (Name & ID)', 'Total Feedback', 'Issues', 'Issue %', 'Severity Score']

        # Color code the severity
        def color_severity(row):
            if row['Severity Score'] >= 10:
                return ['background-color: #ffebee'] * len(row)  # Light red
            elif row['Severity Score'] >= 5:
                return ['background-color: #fff3e0'] * len(row)  # Light orange
            else:
                return [''] * len(row)

        st.dataframe(problem_table.style.apply(color_severity, axis=1), use_container_width=True)

        # Show top issues for problem interpreters
        st.subheader("🔍 Issues by Problem Interpreters")
        for _, interpreter in problem_interpreters.head(5).iterrows():
            with st.expander(f"Issues for {interpreter['interpreter_name']} (ID: {interpreter['InterpreterId']}) - {interpreter['issue_count']} issues"):
                interpreter_issues = df_clean[
                    (df_clean['InterpreterId'] == interpreter['InterpreterId']) &
                    (df_clean['theme'] == 'Issues')
                ]

                for _, issue_row in interpreter_issues.iterrows():
                    category_display = {
                        'audio_quality_poor': '🔊 Poor Audio Quality',
                        'rude_behavior': '😠 Rude Behavior',
                        'translation_errors': '❌ Translation Errors',
                        'call_disconnected': '📞 Call Disconnected',
                        'rushing_interpretation': '⏰ Rushing/Hurried',
                        'interpreter_absent': '❌ Interpreter Absent',
                        'inappropriate_conduct': '⚠️ Inappropriate Conduct',
                        'impatient_behavior': '😤 Impatient Behavior'
                    }.get(issue_row['specific_category'], issue_row['specific_category'])

                    st.write(f"**{category_display}**: \"{issue_row['feedback_clean']}\"")
    else:
        st.success("✅ No major interpreter performance concerns identified!")

    # Show all interpreter stats
    st.subheader("📊 All Interpreter Performance Summary")

    # Filter interpreters with at least 1 feedback
    all_interpreters = interpreter_stats[interpreter_stats['total_feedback'] > 0].copy()
    all_interpreters_display = all_interpreters[['interpreter_info', 'total_feedback', 'issue_count', 'positive_count', 'issue_percentage']].copy()
    all_interpreters_display.columns = ['Interpreter (Name & ID)', 'Total Feedback', 'Issues', 'Positive', 'Issue %']

    st.dataframe(all_interpreters_display, use_container_width=True)

    # Detailed issue breakdown by category
    st.subheader("🔴 Detailed Issues Breakdown")

    specific_categories = {
        'audio_quality_poor': '🔊 Poor Audio Quality',
        'audio_volume_low': '🔉 Low Audio Volume',
        'audio_background_noise': '📢 Background Noise',
        'video_quality_poor': '📹 Poor Video Quality',
        'connection_problems': '🔌 Connection Problems',
        'call_disconnected': '📞 Call Disconnected',
        'technical_difficulties': '⚙️ Technical Difficulties',
        'rude_behavior': '😠 Rude Behavior',
        'impatient_behavior': '😤 Impatient Behavior',
        'inappropriate_conduct': '⚠️ Inappropriate Conduct',
        'rushing_interpretation': '⏰ Rushing/Hurried',
        'translation_errors': '❌ Translation Errors',
        'incomplete_interpretation': '🔄 Incomplete Interpretation',
        'slow_interpretation': '🐌 Slow Interpretation',
        'interpreter_absent': '❌ Interpreter Absent',
        'interpreter_unresponsive': '🔇 Interpreter Unresponsive'
    }

    for category, label in specific_categories.items():
        category_data = df_clean[df_clean['specific_category'] == category]
        if not category_data.empty:
            with st.expander(f"{label} ({len(category_data)} cases)"):
                for _, row in category_data.iterrows():
                    st.write(f"**{row['interpreter_info']}**: \"{row['feedback_clean']}\"")

    # Positive feedback examples
    st.subheader("🟢 Positive Feedback Examples")

    positive_categories = {
        'excellent_service': '⭐ Excellent Service',
        'professional_courteous': '👔 Professional & Courteous',
        'accurate_translation': '✅ Accurate Translation',
        'patient_helpful': '🤝 Patient & Helpful',
        'clear_communication': '💬 Clear Communication',
        'brief_positive': '👍 Brief Positive'
    }

    for category, label in positive_categories.items():
        category_data = df_clean[df_clean['specific_category'] == category]
        if not category_data.empty:
            with st.expander(f"{label} ({len(category_data)} cases)"):
                # Show first 5 examples to keep it manageable
                for _, row in category_data.head(5).iterrows():
                    st.write(f"**{row['interpreter_info']}**: \"{row['feedback_clean']}\"")

# Main Streamlit App
def main():
    st.title("🤖 AI-Powered CSAT Feedback Analysis")
    st.markdown("### Enhanced Medical Interpreter Service Feedback Analyzer with Persistence")

    # Check for existing analysis
    existing_df, existing_interpreter_stats = load_existing_analysis()

    if existing_df is not None:
        # Show analysis timestamp if available
        if 'analysis_timestamp' in existing_df.columns:
            analysis_time = existing_df['analysis_timestamp'].iloc[0]
            st.success(f"📊 **Loaded Previous Analysis** (Generated: {analysis_time})")
        else:
            st.success("📊 **Loaded Previous Analysis**")

        st.info(f"Found {len(existing_df)} analyzed feedback entries from {existing_df['interpreter_name'].nunique()} interpreters")

        # Show visualizations immediately
        st.subheader("📈 Analysis Results (From Saved Data)")
        fig_pie_main, fig_pie_specific, fig_bar = create_visualizations(existing_df)

        # Main overview
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie_main, use_container_width=True)

        with col2:
            st.plotly_chart(fig_pie_specific, use_container_width=True)

        # Issues bar chart
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("🎉 No issues found in the feedback data!")

        # Detailed results
        if existing_interpreter_stats is not None:
            display_detailed_results(existing_df, existing_interpreter_stats)
        else:
            # Recalculate interpreter stats if not available
            interpreter_stats = analyze_interpreter_performance(existing_df)
            display_detailed_results(existing_df, interpreter_stats)

        st.divider()

    # Always show upload option
    st.subheader("📤 Upload New Data or Update Analysis")

    if existing_df is not None:
        st.info("💡 Upload a new CSV file to update the analysis with fresh data")
    else:
        st.info("📋 No previous analysis found. Upload a CSV file to start AI analysis.")

    # Ollama settings
    st.sidebar.header("🤖 AI Model Settings")

    # Model selection
    model_options = ["llama3.2", "llama3.2:1b", "mistral", "phi3", "gemma2", "qwen2.5"]
    selected_model = st.sidebar.selectbox(
        "Select Ollama Model",
        options=model_options,
        index=0,
        help="Smaller models (1b, 3b) are faster but less accurate. Larger models (7b+) are more accurate but slower."
    )

    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="URL where Ollama is running"
    )

    # Initialize analyzer
    try:
        analyzer = OllamaFeedbackAnalyzer(model_name=selected_model, base_url=ollama_url)

        # Test model if user requests
        if st.sidebar.button("🧪 Test Model"):
            analyzer.test_model_simple()

    except Exception as e:
        st.error(f"Failed to initialize Ollama analyzer: {e}")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSAT CSV file",
        type=['csv'],
        help="CSV should contain: ClientFeedback, InterpreterFirstName, InterpreterLastName, InterpreterId columns"
    )

    if uploaded_file is not None:
        # Load and process new data
        with st.spinner("Loading new feedback data..."):
            df = load_and_clean_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Loaded {len(df)} feedback entries from {df['interpreter_name'].nunique()} interpreters")

            # Show a few examples before analysis
            st.subheader("📋 Sample Feedback to Analyze")
            sample_feedback = df[['interpreter_info', 'feedback_clean']].head(3)
            for _, row in sample_feedback.iterrows():
                st.write(f"**{row['interpreter_info']}**: \"{row['feedback_clean']}\"")

            # Check if model is working
            if len(analyzer.available_models) == 0:
                st.error("❌ Cannot proceed - Ollama is not accessible")
                st.stop()

            # Analyze button
            if st.button("🚀 Start Enhanced AI Analysis", type="primary"):

                # Test model first
                if not analyzer.test_model_simple():
                    st.error("❌ Model test failed - cannot proceed with analysis")
                    st.stop()

                with st.spinner("🤖 AI is analyzing feedback with enhanced categorization..."):
                    df_analyzed = analyze_feedback_with_ollama(df, analyzer)

                st.success(f"✅ Enhanced AI analysis complete! Processed {len(df_analyzed)} feedback entries")

                # Analyze interpreter performance
                with st.spinner("👤 Analyzing interpreter performance..."):
                    interpreter_stats = analyze_interpreter_performance(df_analyzed)

                # Save results to CSV
                with st.spinner("💾 Saving analysis results..."):
                    save_analysis_results(df_analyzed, interpreter_stats)

                # Show AI improvements
                st.info("🧠 **Enhanced AI Analysis Active:** Specific issue categorization, interpreter performance tracking, severity scoring")

                # Show visualizations
                st.subheader("📈 Enhanced AI Analysis Results")
                fig_pie_main, fig_pie_specific, fig_bar = create_visualizations(df_analyzed)

                # Main overview
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie_main, use_container_width=True)

                with col2:
                    st.plotly_chart(fig_pie_specific, use_container_width=True)

                # Issues bar chart
                if fig_bar is not None:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("🎉 No issues found in the feedback data!")

                # Detailed results with interpreter analysis
                display_detailed_results(df_analyzed, interpreter_stats)

                # Option to download processed data
                st.subheader("📥 Download Enhanced Analysis Results")

                # Prepare download data with interpreter info
                download_df = df_analyzed[['Months in est_time_Zone', 'InterpreterCompanyName', 'InterpreterId',
                                         'InterpreterFirstName', 'InterpreterLastName', 'TargetLanguage',
                                         'ClientFeedback', 'interpreter_info', 'specific_category', 'theme']].copy()

                csv_download = download_df.to_csv(index=False)
                st.download_button(
                    label="Download Enhanced AI Analysis Results",
                    data=csv_download,
                    file_name="enhanced_ai_csat_analysis_results.csv",
                    mime="text/csv"
                )

    # Show setup instructions
    if existing_df is None:
        st.subheader("🔧 Setup Instructions")

        st.markdown("""
        **Quick Setup (5 minutes):**

        1. **Install Ollama:**
        ```bash
        # Linux/Mac
        curl -fsSL https://ollama.ai/install.sh | sh

        # Windows: Download from https://ollama.ai
        ```

        2. **Start Ollama:**
        ```bash
        ollama serve
        ```

        3. **Install a model (in another terminal):**
        ```bash
        ollama pull llama3.2:1b    # Fast, small model (1.3GB)
        # or
        ollama pull llama3.2       # Better accuracy (2GB)
        # or
        ollama pull mistral        # Good balance (4GB)
        ```

        4. **Verify it's working:**
        ```bash
        ollama list               # Should show your models
        curl http://localhost:11434/api/tags  # Should return JSON
        ```

        **Then upload your CSV file above to start analysis!**
        """)

        st.subheader("🆕 Enhanced Features")

        st.markdown("""
        **💾 Persistent Analysis:**
        - **Auto-Save**: Analysis results automatically saved to CSV
        - **Quick Load**: Previous analysis loads instantly on revisit
        - **Always Fresh**: Upload new data anytime to update analysis

        **🎯 Specific Issue Detection:**
        - **Audio Issues**: Poor quality, low volume, background noise
        - **Video Issues**: Poor quality, frozen video
        - **Connection Issues**: Can't connect, calls disconnected
        - **Behavior Issues**: Rude, impatient, inappropriate conduct
        - **Service Issues**: Translation errors, slow/incomplete interpretation
        - **Availability Issues**: Interpreter absent or unresponsive

        **👤 Interpreter Performance Tracking:**
        - **Problem Identification**: Flags interpreters with concerning patterns
        - **Severity Scoring**: Weights different issue types by impact
        - **Performance Metrics**: Issue percentage, positive feedback ratio
        - **Individual Analysis**: Shows specific issues per interpreter
        """)

if __name__ == "__main__":
    main()