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
        """Create a structured prompt for categorizing feedback"""
        
        prompt = f"""Analyze this customer feedback for a medical interpreter service. Categorize it into ONE of these categories:

ISSUE CATEGORIES:
- connection_issues: Problems connecting to the service
- disconnection_issues: Calls being disconnected or ended prematurely  
- audio_video_issues: Audio quality, noise, volume, video problems
- professionalism_issues: Unprofessional behavior, inappropriate conduct
- accuracy_issues: Translation/interpretation accuracy or speed problems (NOT positive mentions)
- availability_issues: Interpreter not available, not present, not responding
- timing_issues: Rushing, being in a hurry, time-related problems

POSITIVE CATEGORIES:
- excellent_service: Excellent, exceptional, fantastic, amazing work
- professional_behavior: Professional, courteous, skilled, clear communication
- helpful_patient: Helpful, patient, kind, sweet, friendly behavior

OTHER:
- brief_positive: Short positive responses like "good", "great", "thank you"
- other: Unclear feedback or doesn't fit above categories

CRITICAL: Pay attention to context and sentiment!
- "Best translator" = POSITIVE (professional_behavior)
- "Translated correctly" = POSITIVE (professional_behavior)  
- "Interpreter didn't translate" = NEGATIVE (accuracy_issues)

Feedback: "{feedback}"

Respond with ONLY the category name (e.g., "professional_behavior")."""

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
        
        if result == "error":
            return "error"
        elif result.lower() in valid_categories:
            return result.lower()
        else:
            # If LLM returned something unexpected, try to map it intelligently
            result_lower = result.lower()
            
            # Look for keywords in the response
            if any(word in result_lower for word in ['professional', 'behavior']):
                return 'professional_behavior'
            elif any(word in result_lower for word in ['excellent', 'amazing', 'fantastic']):
                return 'excellent_service'
            elif any(word in result_lower for word in ['helpful', 'patient', 'kind']):
                return 'helpful_patient'
            elif any(word in result_lower for word in ['connection', 'connect']):
                return 'connection_issues'
            elif any(word in result_lower for word in ['disconnect', 'hung up']):
                return 'disconnection_issues'
            elif any(word in result_lower for word in ['audio', 'sound', 'noise']):
                return 'audio_video_issues'
            elif any(word in result_lower for word in ['accuracy', 'translate', 'interpret']) and any(word in result_lower for word in ['problem', 'issue', 'bad']):
                return 'accuracy_issues'
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
            time.sleep(0.5)
        
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
        progress_text.text(f"AI analyzing feedback {current}/{total}... ({progress:.0%})")
    
    # Analyze feedback using Ollama
    feedback_list = df['feedback_clean'].tolist()
    categories = analyzer.analyze_feedback_batch(feedback_list, update_progress)
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Add results to dataframe
    df['category'] = categories
    
    # Count errors
    error_count = sum(1 for cat in categories if cat == 'error')
    if error_count > 0:
        st.warning(f"⚠️ {error_count} items could not be analyzed due to API errors")
    
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
    
    # Filter out errors for visualization
    df_clean = df[df['category'] != 'error']
    
    # Overall sentiment breakdown
    theme_counts = df_clean['theme'].value_counts()
    fig_pie = px.pie(
        values=theme_counts.values, 
        names=theme_counts.index,
        title="AI-Powered Feedback Analysis Results",
        color_discrete_map={'Issues': '#ff6b6b', 'Positive': '#51cf66', 'Other': '#ffd43b'}
    )
    
    # Issue breakdown by category
    issue_df = df_clean[df_clean['theme'] == 'Issues']
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
    
    # Filter out errors for analysis
    df_clean = df[df['category'] != 'error']
    
    # Summary statistics
    total_feedback = len(df)
    total_analyzed = len(df_clean)
    issues_count = len(df_clean[df_clean['theme'] == 'Issues'])
    positive_count = len(df_clean[df_clean['theme'] == 'Positive'])
    other_count = len(df_clean[df_clean['theme'] == 'Other'])
    error_count = len(df[df['category'] == 'error'])
    
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
    
    # Show AI categorization examples
    st.subheader("🤖 AI Categorization Examples")
    
    # Show some examples of how AI categorized key cases
    examples_to_show = [
        "Translating medical terminology correctly",
        "Best translator ive had. Translated everything clearly and efficiently.",
        "Very patient and skilled",
        "interpreter did not interpret",
        "call cut off",
        "very rude"
    ]
    
    for example_text in examples_to_show:
        matching_rows = df[df['feedback_clean'].str.contains(example_text, case=False, na=False)]
        if not matching_rows.empty:
            actual_category = matching_rows.iloc[0]['category']
            actual_theme = matching_rows.iloc[0]['theme']
            
            # Color code based on whether it looks correct
            if actual_theme == "Positive" and any(word in example_text.lower() for word in ['best', 'correctly', 'patient', 'skilled']):
                status_icon = "✅"
            elif actual_theme == "Issues" and any(word in example_text.lower() for word in ['did not', 'cut off', 'rude']):
                status_icon = "✅"
            elif actual_category == "error":
                status_icon = "❌"
            else:
                status_icon = "🤔"
            
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
        category_data = df_clean[df_clean['category'] == category]
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
        category_data = df_clean[df_clean['category'] == category]
        if not category_data.empty:
            with st.expander(f"{label} ({len(category_data)} cases)"):
                # Show first 5 examples to keep it manageable
                for _, row in category_data.head(5).iterrows():
                    st.write(f"• \"{row['feedback_clean']}\"")

# Main Streamlit App
def main():
    st.title("🤖 AI-Powered CSAT Feedback Analysis")
    st.markdown("### Medical Interpreter Service Feedback Analyzer using Ollama LLM")
    
    st.markdown("""
    This enhanced tool uses **Large Language Model (LLM) AI** via Ollama to understand context, sentiment, 
    and nuanced meaning in feedback - solving the keyword-based approach limitations.
    
    **Key Improvements:**
    - 🧠 **Context-aware**: Understands "best translator" vs "interpreter problems"
    - 🎯 **Sentiment analysis**: Distinguishes praise from complaints
    - 🔧 **Handles ambiguity**: Processes complex, multi-topic feedback
    - 📊 **More accurate**: Reduces false positives/negatives
    """)
    
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
            
            # Check if model is working
            if len(analyzer.available_models) == 0:
                st.error("❌ Cannot proceed - Ollama is not accessible")
                st.stop()
            
            # Analyze button
            if st.button("🚀 Start AI Analysis", type="primary"):
                
                # Test model first
                if not analyzer.test_model_simple():
                    st.error("❌ Model test failed - cannot proceed with analysis")
                    st.stop()
                
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
        # Show setup instructions while waiting for upload
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
        
        st.subheader("🧠 Why AI Analysis is Better")
        
        st.markdown("""
        **❌ Keyword-Based Problems (Old Approach):**
        - "Best translator" → Wrongly detected as "accuracy issue"
        - "Translating correctly" → Wrongly detected as "problem"
        - Cannot understand context or sentiment
        - Many false positives and negatives
        
        **✅ AI-Powered Solutions (New Approach):**
        - **Context Understanding**: "Best translator" → Correctly categorized as positive feedback ✓
        - **Sentiment Analysis**: "Translating correctly" → Correctly categorized as praise ✓
        - **Nuanced Processing**: Handles complex, multi-topic feedback
        - **Reduced Errors**: Dramatically more accurate categorization
        
        **Example AI Categories Detected:**
        
        **🔴 Issue Categories:**
        - Connection/Disconnection Problems
        - Audio/Video Quality Issues  
        - Professionalism & Behavior Issues
        - Accuracy/Speed Problems (context-aware)
        - Availability & Timing Issues
        
        **🟢 Positive Categories:**
        - Excellent Service Recognition
        - Professional Behavior Praise
        - Helpful/Patient Feedback
        - Brief Positive Responses
        """)

if __name__ == "__main__":
    main()
