import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz, process
import string

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Enhanced CSAT Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

class RobustFeedbackCategorizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
        
        # Enhanced keyword dictionaries with variations, misspellings, and synonyms
        self.keywords = {
            'connection_issues': {
                'primary': ['connect', 'connection', 'connecting'],
                'variations': ['conect', 'conectivity', 'conection'],
                'phrases': ['not connecting', 'no connection', 'not able to connect', 
                           'unable to connect', 'connection failed', 'failed to connect']
            },
            'disconnection_issues': {
                'primary': ['disconnect', 'disconnected', 'hang up', 'hung up', 'cut off'],
                'variations': ['disconect', 'dissconnect', 'hanged up'],
                'phrases': ['call ended', 'ended call', 'randomly ended', 'call cut off',
                           'ended randomly', 'keep getting disconnected']
            },
            'audio_video_issues': {
                'primary': ['audio', 'sound', 'hear', 'noise', 'volume', 'loud', 'quiet'],
                'variations': ['hearring', 'nois', 'volum'],
                'phrases': ['black screen', 'no audio', 'no sound', 'background noise',
                           'difficult to hear', 'hard time hearing', 'music playing',
                           'radio playing', 'dog barking', 'a lot of noise']
            },
            'professionalism_issues': {
                'primary': ['rude', 'unprofessional', 'aggressive', 'inappropriate', 'awful'],
                'variations': ['profesional', 'proffessional', 'awfull', 'aweful'],
                'phrases': ['not wearing a shirt', 'wasn\'t wearing a shirt', 'very rude',
                           'extremely unprofessional', 'just a chair', 'person not there']
            },
            'accuracy_issues': {
                'primary': ['interpret', 'translate', 'listening', 'understand'],
                'variations': ['intepreter', 'interperter', 'interprter', 'translat'],
                'phrases': ['not listening', 'did not interpret', 'takes too long to translate',
                           'not interpreting', 'making noise', 'did not understand']
            },
            'availability_issues': {
                'primary': ['answer', 'available', 'there', 'present'],
                'variations': ['availble', 'ther'],
                'phrases': ['never answered', 'no answer', 'person not there',
                           'left before', 'not there', 'just a chair']
            },
            'timing_issues': {
                'primary': ['anxious', 'rush', 'hurry', 'quick'],
                'variations': ['anxius', 'anxyous'],
                'phrases': ['anxious to end', 'rushed the call', 'seemed in a hurry']
            },
            'excellent_service': {
                'primary': ['excellent', 'exceptional', 'fantastic', 'awesome', 'amazing', 'wonderful'],
                'variations': ['exellent', 'excelent', 'awsome', 'amazin', 'wonderfull'],
                'phrases': ['excellent service', 'exceptional job', 'fantastic job',
                           'wonderful job', 'amazing job']
            },
            'professional_behavior': {
                'primary': ['professional', 'courteous', 'skilled', 'clear', 'polite'],
                'variations': ['profesional', 'proffessional', 'skiled', 'cler'],
                'phrases': ['well spoken', 'very professional', 'very skilled',
                           'very clear', 'spoke clearly']
            },
            'helpful_patient': {
                'primary': ['helpful', 'patient', 'kind', 'sweet', 'friendly'],
                'variations': ['helpfull', 'pacient', 'patiant', 'freindly'],
                'phrases': ['very helpful', 'very patient', 'very kind',
                           'very sweet', 'very friendly']
            }
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing with stemming and normalization"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = str(text).lower().strip()
        
        # Remove punctuation except apostrophes (for contractions)
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def stem_keywords(self, keyword_list):
        """Stem keywords for better matching"""
        stemmed = []
        for keyword in keyword_list:
            words = keyword.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            stemmed.append(' '.join(stemmed_words))
        return stemmed
    
    def fuzzy_match_keywords(self, text, keywords, threshold=75):
        """Use fuzzy matching to find similar keywords"""
        words = text.split()
        
        # Check individual words and phrases
        for keyword in keywords:
            # Direct substring check
            if keyword in text:
                return True
            
            # Fuzzy match for individual words
            for word in words:
                if fuzz.ratio(keyword, word) >= threshold:
                    return True
            
            # Fuzzy match for phrases (if keyword has multiple words)
            if ' ' in keyword:
                keyword_words = keyword.split()
                for i in range(len(words) - len(keyword_words) + 1):
                    phrase = ' '.join(words[i:i+len(keyword_words)])
                    if fuzz.ratio(keyword, phrase) >= threshold:
                        return True
        
        return False
    
    def check_category_match(self, text, category_keywords, fuzzy_threshold=75):
        """Check if text matches any keywords in a category using multiple techniques"""
        preprocessed_text = self.preprocess_text(text)
        
        # Stem the preprocessed text
        words = preprocessed_text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        stemmed_text = ' '.join(stemmed_words)
        
        # Check primary keywords with exact and fuzzy matching
        all_keywords = (category_keywords.get('primary', []) + 
                       category_keywords.get('variations', []) + 
                       category_keywords.get('phrases', []))
        
        # 1. Direct substring matching (fastest)
        for keyword in all_keywords:
            if keyword in preprocessed_text:
                return True
        
        # 2. Stemmed keyword matching
        stemmed_keywords = self.stem_keywords(all_keywords)
        for keyword in stemmed_keywords:
            if keyword in stemmed_text:
                return True
        
        # 3. Fuzzy matching (for misspellings)
        if self.fuzzy_match_keywords(preprocessed_text, all_keywords, fuzzy_threshold):
            return True
        
        # 4. Partial ratio for substring variations
        for keyword in all_keywords:
            if fuzz.partial_ratio(keyword, preprocessed_text) >= fuzzy_threshold + 10:
                return True
        
        return False
    
    def categorize_feedback(self, feedback_text, fuzzy_threshold=75):
        """Enhanced categorization with hierarchical priority and fuzzy matching"""
        if not feedback_text or pd.isna(feedback_text):
            return 'other'
        
        # Hierarchical categorization (technical issues first, then service, then positive)
        priority_order = [
            'connection_issues', 'disconnection_issues', 'audio_video_issues',
            'professionalism_issues', 'accuracy_issues', 'availability_issues', 
            'timing_issues', 'excellent_service', 'professional_behavior', 
            'helpful_patient'
        ]
        
        for category in priority_order:
            if self.check_category_match(feedback_text, self.keywords[category], fuzzy_threshold):
                return category
        
        # Brief positive responses (fallback)
        preprocessed = self.preprocess_text(feedback_text)
        if (any(word in preprocessed for word in ['good', 'great', 'thank']) and 
            len(preprocessed) < 50):
            return 'brief_positive'
        
        return 'other'

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

def analyze_feedback(df, fuzzy_threshold=75):
    """Perform the main analysis with enhanced categorization"""
    categorizer = RobustFeedbackCategorizer()
    
    # Categorize all feedback
    df['category'] = df['feedback_clean'].apply(
        lambda x: categorizer.categorize_feedback(x, fuzzy_threshold)
    )
    
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
        title="Overall Feedback Distribution",
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
            title="Issues by Category",
            labels={'x': 'Number of Cases', 'y': 'Issue Type'},
            color=issue_counts.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(showlegend=False)
    else:
        fig_bar = None
    
    return fig_pie, fig_bar

def display_detailed_results(df):
    """Display detailed breakdown tables"""
    
    # Summary statistics
    total_feedback = len(df)
    issues_count = len(df[df['theme'] == 'Issues'])
    positive_count = len(df[df['theme'] == 'Positive'])
    other_count = len(df[df['theme'] == 'Other'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", total_feedback)
    with col2:
        st.metric("Issues Identified", f"{issues_count} ({issues_count/total_feedback*100:.0f}%)")
    with col3:
        st.metric("Positive Feedback", f"{positive_count} ({positive_count/total_feedback*100:.0f}%)")
    with col4:
        st.metric("Other/Unclear", f"{other_count} ({other_count/total_feedback*100:.0f}%)")
    
    # Detailed issue breakdown
    st.subheader("🔴 Issues Breakdown")
    
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
    st.subheader("🟢 Positive Feedback Examples")
    
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
    """Generate actionable recommendations"""
    
    st.subheader("🎯 Actionable Insights & Recommendations")
    
    total_feedback = len(df)
    issues_df = df[df['theme'] == 'Issues']
    
    # Calculate technical issues percentage
    technical_issues = ['connection_issues', 'disconnection_issues', 'audio_video_issues']
    technical_count = len(issues_df[issues_df['category'].isin(technical_issues)])
    
    st.markdown("### Priority 1: Technical Infrastructure (CRITICAL)")
    st.error(f"**{technical_count} cases ({technical_count/total_feedback*100:.0f}% of all feedback) are technical issues**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Immediate Actions:**
        - Audit connection infrastructure
        - Implement connection monitoring
        - Create audio quality standards
        - Establish environment guidelines for interpreters
        """)
    
    with col2:
        # Show top technical issues
        tech_breakdown = issues_df[issues_df['category'].isin(technical_issues)]['category'].value_counts()
        if not tech_breakdown.empty:
            st.markdown("**Technical Issues Breakdown:**")
            for issue, count in tech_breakdown.items():
                issue_name = issue.replace('_', ' ').title()
                st.write(f"• {issue_name}: {count} cases")
    
    # Service quality recommendations
    service_issues = ['professionalism_issues', 'accuracy_issues', 'availability_issues', 'timing_issues']
    service_count = len(issues_df[issues_df['category'].isin(service_issues)])
    
    if service_count > 0:
        st.markdown("### Priority 2: Service Quality Control")
        st.warning(f"**{service_count} cases related to interpreter performance**")
        st.markdown("""
        **Recommended Actions:**
        - Implement interpreter training program
        - Create session completion protocols
        - Establish professionalism guidelines
        - Regular performance monitoring
        """)
    
    # Positive feedback insights
    positive_count = len(df[df['theme'] == 'Positive'])
    st.markdown("### 🟢 Success Factors to Leverage")
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
    st.title("📊 Enhanced CSAT Feedback Analysis Tool")
    st.markdown("### Medical Interpreter Service Feedback Analyzer with Fuzzy Matching")
    
    st.markdown("""
    This enhanced tool uses advanced text processing techniques including **fuzzy matching**, **stemming**, 
    and **expanded keyword dictionaries** to identify common issues and positive trends even with misspellings 
    and variations in wording.
    """)
    
    # Fuzzy matching threshold setting
    st.sidebar.header("⚙️ Advanced Settings")
    fuzzy_threshold = st.sidebar.slider(
        "Fuzzy Matching Sensitivity", 
        min_value=60, max_value=90, value=75,
        help="Higher values = more strict matching. Lower values = catches more variations but may have false positives."
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSAT CSV file", 
        type=['csv'],
        help="CSV should contain a 'ClientFeedback' column with client feedback text"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing feedback data with enhanced algorithms..."):
            df = load_and_clean_data(uploaded_file)
            
        if df is not None:
            # Perform analysis
            df_analyzed = analyze_feedback(df, fuzzy_threshold)
            
            # Display results
            st.success(f"✅ Analyzed {len(df_analyzed)} feedback entries with enhanced fuzzy matching")
            
            # Show improvements detected
            st.info("🔧 **Enhanced Features Active:** Fuzzy matching for misspellings, stemming for word variations, expanded keyword dictionaries")
            
            # Show visualizations
            st.subheader("📈 Analysis Overview")
            fig_pie, fig_bar = create_visualizations(df_analyzed)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if fig_bar is not None:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No issues found in the feedback data!")
            
            # Summary metrics and detailed breakdown
            display_detailed_results(df_analyzed)
            
            # Actionable insights
            generate_actionable_insights(df_analyzed)
            
            # Option to download processed data
            st.subheader("📥 Download Results")
            csv_download = df_analyzed.to_csv(index=False)
            st.download_button(
                label="Download Analyzed Data as CSV",
                data=csv_download,
                file_name="enhanced_csat_analysis_results.csv",
                mime="text/csv"
            )
    
    else:
        # Show methodology while waiting for upload
        st.subheader("🔧 Enhanced Analysis Methodology")
        
        st.markdown("""
        **This tool now includes advanced robustness features:**
        
        **1. Fuzzy String Matching**
        - Handles misspellings like "EXELLENT" → "EXCELLENT"
        - Uses Levenshtein distance algorithm
        - Configurable sensitivity threshold
        
        **2. Text Preprocessing & Stemming**
        - Reduces words to root forms: "helpful"/"helping" → "help"
        - Removes stop words and normalizes text
        - Handles contractions and punctuation
        
        **3. Expanded Keyword Dictionaries**
        - Primary keywords + common variations + misspellings
        - Synonyms: "translator" ↔ "interpreter"
        - Phrase matching: "radio playing", "wasn't wearing a shirt"
        
        **4. Multi-Level Matching Strategy**
        - Direct substring matching (fastest)
        - Stemmed keyword matching
        - Fuzzy matching for misspellings
        - Partial ratio matching for phrases
        
        **5. Hierarchical Classification**
        - Technical issues prioritized first
        - Service issues second
        - Positive feedback last
        
        **Issue Categories Detected:**
        - 🔴 Connection Issues (with variants like "conect", "conectivity")
        - 🔴 Disconnection Problems (including "randomly ended", "hung up")
        - 🔴 Audio/Video Quality (including "radio playing", "hard time hearing")
        - 🔴 Professionalism Issues (including "wasn't wearing a shirt")
        - 🔴 Accuracy/Speed Problems (including misspellings of "interpreter")
        - 🔴 Availability Issues (including "just a chair", "person not there")
        - 🟢 Excellent Service (including "EXELLENT", "awsome")
        - 🟢 Professional Behavior (including "profesional", "well spoken")
        - 🟢 Helpful/Patient Feedback (including "helpfull", "pacient")
        """)

if __name__ == "__main__":
    main()
