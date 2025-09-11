import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="AI-Powered CSAT Feedback Analyzer",
    page_icon="🤖",
    layout="wide"
)

class OllamaFeedbackAnalyzer:
    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                st.sidebar.success(f"✅ Ollama connected! Available models: {', '.join(available_models)}")
                
                if self.model_name not in [m.split(':')[0] for m in available_models]:
                    st.sidebar.warning(f"⚠️ Model '{self.model_name}' not found. Available: {available_models}")
            else:
                st.sidebar.error("❌ Ollama is running but API not responding correctly")
        except requests.exceptions.RequestException:
            st.sidebar.error("❌ Cannot connect to Ollama. Make sure it's running on http://localhost:11434")
            st.sidebar.info("💡 Start Ollama with: `ollama serve` and install a model like `ollama pull mistral`")
    
    def query_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Send a query to Ollama with retry logic"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent categorization
                "top_p": 0.9,
                "num_predict": 50    # Limit response length
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    json=payload, 
                    timeout=30,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    st.error(f"Ollama API error: {response.status_code}")
                    return "error"
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to connect to Ollama after {max_retries} attempts: {e}")
                    return "error"
                time.sleep(1)  # Wait before retry
        
        return "error"
    
    def create_categorization_prompt(self, feedback: str) -> str:
        """Create a structured prompt for categorizing feedback"""
        
        prompt = f"""You are analyzing customer feedback for a medical interpreter service. Your job is to categorize the feedback into one of these exact categories:

ISSUE CATEGORIES (problems/complaints):
- connection_issues: Problems connecting to the service
- disconnection_issues: Calls being disconnected or ended prematurely  
- audio_video_issues: Audio quality, noise, volume, video problems
- professionalism_issues: Unprofessional behavior, inappropriate conduct
- accuracy_issues: Translation/interpretation accuracy or speed problems
- availability_issues: Interpreter not available, not present, not responding
- timing_issues: Rushing, being in a hurry, time-related problems

POSITIVE CATEGORIES (compliments/praise):
- excellent_service: Excellent, exceptional, fantastic, amazing work
- professional_behavior: Professional, courteous, skilled, clear communication
- helpful_patient: Helpful, patient, kind, sweet, friendly behavior

OTHER:
- brief_positive: Short positive responses like "good", "great", "thank you"
- other: Unclear feedback or doesn't fit above categories

CRITICAL: Pay attention to context and sentiment. Words like "translate" or "interpreter" can be positive when used in praise (e.g., "best translator", "translated correctly") or negative when describing problems.

Feedback to categorize: "{feedback}"

Respond with ONLY the category name (e.g., "professional_behavior" or "connection_issues"). No explanations."""

        return prompt
    
    def categorize_single_feedback(self, feedback: str) -> str:
        """Categorize a single piece of feedback using Ollama"""
        if not feedback or pd.isna(feedback) or feedback.strip() == "":
            return "other"
        
        prompt = self.create_categorization_prompt(feedback)
        result = self.query_ollama(prompt)
        
        # Validate the result is a known category
        valid_categories = {
            'connection_issues', 'disconnection_issues', 'audio_video_issues',
            'professionalism_issues', 'accuracy_issues', 'availability_issues', 
            'timing_issues', 'excellent_service', 'professional_behavior', 
            'helpful_patient', 'brief_positive', 'other'
        }
        
        if result.lower() in valid_categories:
            return result.lower()
        elif result == "error":
            return "error"
        else:
            # If LLM returned something unexpected, try to map it
            result_lower = result.lower()
            if any(word in result_lower for word in ['connection', 'connect']):
                return 'connection_issues'
            elif any(word in result_lower for word in ['disconnect', 'hung up', 'cut off']):
                return 'disconnection_issues'
            elif any(word in result_lower for word in ['audio', 'sound', 'noise']):
                return 'audio_video_issues'
            elif any(word in result_lower for word in ['professional']):
                return 'professional_behavior'
            elif any(word in result_lower for word in ['excellent', 'amazing']):
                return 'excellent_service'
            else:
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
            time.sleep(0.1)
        
        return results

def load_and_clean_data(uploaded_file):
    """Load and clean the CSV data"""
    df = pd.read_csv(uploaded_file)
    
    # Clean the feedback column
    if 'ClientFeedback' in df.columns:
        df['feedback_clean'] = df['ClientFeedback'].fillna('').str.strip()
        df = df[df['feedback_clean'] != '']  # Remove empty feedback
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
        progress_text.text(f"Analyzing feedback {current}/{total}...")
    
    # Analyze feedback using Ollama
    feedback_list = df['feedback_clean'].tolist()
    categories = analyzer.analyze_feedback_batch(feedback_list, update_progress)
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Add results to dataframe
    df['category'] = categories
    
    # Group categories into broader themes
    issue_categories = ['connection_issues', 'disconnection_issues', 'audio_video_issues', 
                       'professionalism_issues', 'accuracy_issues', 'availability_issues', 'timing_issues']
    positive_categories = ['excellent_service', 'professional_behavior', 'helpful_patient', 'brief_positive']
    
    df['theme'] = df['category'].apply(lambda x: 'Issues' if x in issue_categories 
                                     else 'Positive' if x in positive_categories 
                                     else 'Other')
    
    return df

def create_visualizations(df):
    """Create charts and visualizations"""
    
    # Overall sentiment breakdown
    theme_counts = df['theme'].value_counts()
    fig_pie = px.pie(
        values=theme_counts.values, 
        names=theme_counts.index,
        title="AI-Powered Feedback Analysis Results",
        color_discrete_map={'Issues': '#ff6b6b', 'Positive': '#51cf66', 'Other': '#ffd43b'}
    )
    
    # Issue breakdown by category
    issue_df = df[df['theme'] == 'Issues']
    if not issue_df.empty:
        issue_counts = issue_df['category'].value_counts()
        
        # Clean up category names for display
        category_labels = {
            'connection_issues': 'Connection Issues',
            'disconnection_issues': 'Disconnection Issues', 
            'audio_video_issues': 'Audio/Video Issues',
            'professionalism_issues': 'Professionalism Issues',
            'accuracy_issues': 'Accuracy/Speed Issues',
            'availability_issues': 'Availability Issues',
            'timing_issues': 'Timing Issues'
        }
        
        display_names = [category_labels.get(cat, cat) for cat in issue_counts.index]
        
        fig_bar = px.bar(
            x=issue_counts.values,
            y=display_names,
            orientation='h',
            title="Issues Identified by AI Analysis",
            labels={'x': 'Number of Cases', 'y': 'Issue Type'},
            color=issue_counts.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(showlegend=False)
    else:
        fig_bar = None
    
    return fig_pie, fig_bar

def display_detailed_results(df):
    """Display detailed breakdown tables with AI insights"""
    
    # Summary statistics
    total_feedback = len(df)
    issues_count = len(df[df['theme'] == 'Issues'])
    positive_count = len(df[df['theme'] == 'Positive'])
    other_count = len(df[df['theme'] == 'Other'])
    error_count = len(df[df['category'] == 'error'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("Issues Identified", f"{issues_count} ({issues_count/total_feedback*100:.0f}%)")
    with col3:
        st.metric("Positive Feedback", f"{positive_count} ({positive_count/total_feedback*100:.0f}%)")
    with col4:
        st.metric("Other/Unclear", f"{other_count} ({other_count/total_feedback*100:.0f}%)")
    with col5:
        if error_count > 0:
            st.metric("Analysis Errors", error_count, delta="⚠️")
    
    # Show AI categorization examples
    st.subheader("🤖 AI Categorization Examples")
    
    # Show some examples of how AI categorized ambiguous cases
    examples_to_show = [
        ("Translating medical terminology correctly", "Should be positive"),
        ("Best translator ive had. Translated everything clearly and efficiently.", "Should be positive"),
        ("interpreter did not interpret", "Should be accuracy issue"),
        ("Very patient and skilled", "Should be positive")
    ]
    
    for example_text, expected in examples_to_show:
        matching_rows = df[df['feedback_clean'].str.contains(example_text, case=False, na=False)]
        if not matching_rows.empty:
            actual_category = matching_rows.iloc[0]['category']
            actual_theme = matching_rows.iloc[0]['theme']
            
            # Determine if categorization looks correct
            is_correct = (
                (expected.startswith("Should be positive") and actual_theme == "Positive") or
                (expected.startswith("Should be accuracy") and actual_category == "accuracy_issues")
            )
            
            status_icon = "✅" if is_correct else "❌"
            st.write(f"{status_icon} **\"{example_text}\"** → `{actual_category}` ({actual_theme})")
    
    # Detailed issue breakdown
    st.subheader("🔴 Issues Breakdown (AI-Detected)")
    
    issue_categories = {
        'connection_issues': 'Connection Issues',
        'disconnection_issues': 'Disconnection Issues', 
        'audio_video_issues': 'Audio/Video Issues',
        'professionalism_issues': 'Professionalism Issues',
        'accuracy_issues': 'Accuracy/Speed Issues',
        'availability_issues': 'Availability Issues',
        'timing_issues': 'Timing Issues'
    }
    
    for category, label in issue_categories.items():
        category_data = df[df['category'] == category]
        if not category_data.empty:
            with st.expander(f"{label} ({len(category_data)} cases)"):
                for _, row in category_data.iterrows():
                    st.write(f"• \"{row['feedback_clean']}\"")
    
    # Positive feedback examples
    st.subheader("🟢 Positive Feedback (AI-Detected)")
    
    positive_categories = {
        'excellent_service': 'Excellent Service',
        'professional_behavior': 'Professional Behavior',
        'helpful_patient': 'Helpful/Patient',
        'brief_positive': 'Brief Positive'
    }
    
    for category, label in positive_categories.items():
        category_data = df[df['category'] == category]
        if not category_data.empty:
            with st.expander(f"{label} ({len(category_data)} cases)"):
                # Show first 5 examples to keep it manageable
                for _, row in category_data.head(5).iterrows():
                    st.write(f"• \"{row['feedback_clean']}\"")

def generate_actionable_insights(df):
    """Generate actionable recommendations based on AI analysis"""
    
    st.subheader("🎯 AI-Generated Insights & Recommendations")
    
    total_feedback = len(df)
    issues_df = df[df['theme'] == 'Issues']
    
    # Calculate technical issues percentage
    technical_issues = ['connection_issues', 'disconnection_issues', 'audio_video_issues']
    technical_count = len(issues_df[issues_df['category'].isin(technical_issues)])
    
    st.markdown("### Priority 1: Technical Infrastructure (AI-Identified)")
    if technical_count > 0:
        st.error(f"**{technical_count} cases ({technical_count/total_feedback*100:.0f}% of all feedback) are technical issues**")
    else:
        st.success("**No significant technical issues detected by AI analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **AI-Recommended Actions:**
        - Audit connection infrastructure
        - Implement real-time connection monitoring
        - Create audio quality standards
        - Establish environment guidelines for interpreters
        """)
    
    with col2:
        # Show top technical issues
        if technical_count > 0:
            tech_breakdown = issues_df[issues_df['category'].isin(technical_issues)]['category'].value_counts()
            st.markdown("**Technical Issues Breakdown:**")
            for issue, count in tech_breakdown.items():
                issue_name = issue.replace('_', ' ').title()
                st.write(f"• {issue_name}: {count} cases")
    
    # Service quality recommendations
    service_issues = ['professionalism_issues', 'accuracy_issues', 'availability_issues', 'timing_issues']
    service_count = len(issues_df[issues_df['category'].isin(service_issues)])
    
    if service_count > 0:
        st.markdown("### Priority 2: Service Quality Control (AI-Identified)")
        st.warning(f"**{service_count} cases related to interpreter performance**")
        st.markdown("""
        **AI-Recommended Actions:**
        - Implement interpreter training program
        - Create session completion protocols
        - Establish professionalism guidelines
        - Regular performance monitoring
        """)
    
    # Positive feedback insights
    positive_count = len(df[df['theme'] == 'Positive'])
    st.markdown("### 🟢 Success Factors to Leverage (AI-Identified)")
    st.success(f"**{positive_count} positive feedback cases ({positive_count/total_feedback*100:.0f}%) highlight key success factors**")
    
    # Extract most common positive themes
    positive_df = df[df['theme'] == 'Positive']
    if not positive_df.empty:
        positive_breakdown = positive_df['category'].value_counts()
        st.markdown("**Top Success Factors:**")
        for category, count in positive_breakdown.head(3).items():
            category_name = category.replace('_', ' ').title()
            st.write(f"• {category_name}: {count} mentions")

# Main Streamlit App
def main():
    st.title("🤖 AI-Powered CSAT Feedback Analysis")
    st.markdown("### Medical Interpreter Service Feedback Analyzer using Ollama LLM")
    
    st.markdown("""
    This enhanced tool uses **Large Language Model (LLM) AI** via Ollama to understand context, sentiment, 
    and nuanced meaning in feedback - solving the limitations of keyword-based approaches.
    
    **Key Improvements:**
    - 🧠 **Context-aware**: Understands "best translator" vs "interpreter problems"
    - 🎯 **Sentiment analysis**: Distinguishes praise from complaints
    - 🔧 **Handles ambiguity**: Processes complex, multi-topic feedback
    - 📊 **More accurate categorization**: Reduces false positives/negatives
    """)
    
    # Ollama settings
    st.sidebar.header("🤖 AI Model Settings")
    
    model_options = ["mistral", "llama2", "codellama", "phi", "neural-chat"]
    selected_model = st.sidebar.selectbox(
        "Select Ollama Model", 
        options=model_options,
        index=0,
        help="Make sure the selected model is installed: `ollama pull <model-name>`"
    )
    
    ollama_url = st.sidebar.text_input(
        "Ollama URL", 
        value="http://localhost:11434",
        help="URL where Ollama is running"
    )
    
    # Initialize analyzer
    try:
        analyzer = OllamaFeedbackAnalyzer(model_name=selected_model, base_url=ollama_url)
    except Exception as e:
        st.error(f"Failed to initialize Ollama analyzer: {e}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSAT CSV file", 
        type=['csv'],
        help="CSV should contain a 'ClientFeedback' column with client feedback text"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading feedback data..."):
            df = load_and_clean_data(uploaded_file)
            
        if df is not None:
            st.success(f"✅ Loaded {len(df)} feedback entries")
            
            # Show a few examples before analysis
            st.subheader("📋 Sample Feedback to Analyze")
            sample_feedback = df['feedback_clean'].head(3).tolist()
            for i, feedback in enumerate(sample_feedback, 1):
                st.write(f"{i}. \"{feedback}\"")
            
            # Analyze button
            if st.button("🚀 Start AI Analysis", type="primary"):
                
                with st.spinner("🤖 AI is analyzing feedback using Ollama LLM..."):
                    df_analyzed = analyze_feedback_with_ollama(df, analyzer)
                
                st.success(f"✅ AI analysis complete! Processed {len(df_analyzed)} feedback entries")
                
                # Show AI improvements
                st.info("🧠 **AI Analysis Active:** Context-aware categorization, sentiment analysis, semantic understanding")
                
                # Show visualizations
                st.subheader("📈 AI Analysis Results")
                fig_pie, fig_bar = create_visualizations(df_analyzed)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    if fig_bar is not None:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("🎉 No issues found in the feedback data!")
                
                # Summary metrics and detailed breakdown
                display_detailed_results(df_analyzed)
                
                # Actionable insights
                generate_actionable_insights(df_analyzed)
                
                # Option to download processed data
                st.subheader("📥 Download AI Analysis Results")
                csv_download = df_analyzed.to_csv(index=False)
                st.download_button(
                    label="Download AI-Analyzed Data as CSV",
                    data=csv_download,
                    file_name="ai_csat_analysis_results.csv",
                    mime="text/csv"
                )
    
    else:
        # Show methodology while waiting for upload
        st.subheader("🧠 AI Analysis Methodology")
        
        st.markdown("""
        **How AI-Powered Analysis Solves Context Problems:**
        
        **❌ Keyword-Based Problems:**
        - "Best translator" → Detected as "accuracy issue" (wrong!)
        - "Translating correctly" → Detected as "problem" (wrong!)
        - Can't understand context or sentiment
        
        **✅ AI-Powered Solutions:**
        - **Context Understanding**: "Best translator" → Positive feedback ✓
        - **Sentiment Analysis**: "Translating correctly" → Positive feedback ✓
        - **Nuanced Processing**: Handles complex, multi-topic feedback
        - **Reduced False Positives**: More accurate categorization
        
        **AI Categories Detected:**
        
        **🔴 Issue Categories:**
        - Connection Issues (e.g., "couldn't connect")
        - Disconnection Problems (e.g., "call cut off", "hung up") 
        - Audio/Video Quality (e.g., "background noise", "hard to hear")
        - Professionalism Issues (e.g., "very rude", "inappropriate")
        - Accuracy/Speed Problems (e.g., "didn't interpret", "too slow")
        - Availability Issues (e.g., "not there", "left early")
        - Timing Issues (e.g., "rushed the call")
        
        **🟢 Positive Categories:**
        - Excellent Service (e.g., "fantastic", "amazing job")
        - Professional Behavior (e.g., "very professional", "skilled")
        - Helpful/Patient (e.g., "very patient", "helpful")
        - Brief Positive (e.g., "good", "great", "thank you")
        
        **🤖 AI Advantages:**
        - Understands context and sentiment
        - Handles misspellings naturally
        - Processes complex sentences
        - Distinguishes praise from complaints
        - Adapts to various writing styles
        """)
        
        st.markdown("""
        **Prerequisites:**
        1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
        2. Start Ollama: `ollama serve`
        3. Install a model: `ollama pull mistral`
        """)

if __name__ == "__main__":
    main()
