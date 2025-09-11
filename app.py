import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="CSAT Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)

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

def categorize_feedback(feedback_text):
    """Categorize feedback using keyword matching with hierarchical priority"""
    feedback = feedback_text.lower()
    
    # Define keyword dictionaries (same as analysis)
    keywords = {
        'connection_issues': ['not connecting', 'no connection', 'black screen', 'no audio', 'no sound', 'never answered', 'no answer'],
        'disconnection_issues': ['cut off', 'hang up', 'hung up', 'disconnected', 'ended call'],
        'audio_video_issues': ['background noise', 'difficult to hear', 'loud', 'music playing', 'radio playing', 'dog barking', 'noise'],
        'professionalism_issues': ['rude', 'aggressive', 'unprofessional', "wasn't wearing a shirt", 'inappropriate', 'awful'],
        'accuracy_issues': ['not listening', 'did not interpret', 'takes too long to translate', 'making so much noise'],
        'availability_issues': ['left before', 'person not there', 'just a chair'],
        'timing_issues': ['anxious to end'],
        'excellent_service': ['excellent', 'exceptional', 'fantastic', 'awesome', 'amazing', 'wonderful'],
        'professional_behavior': ['professional', 'courteous', 'patient', 'skilled', 'clear', 'well spoken'],
        'helpful_patient': ['helpful', 'kind', 'sweet', 'friendly']
    }
    
    # Hierarchical categorization (technical issues first, then service, then positive)
    for category, words in keywords.items():
        if any(word in feedback for word in words):
            return category
    
    # Brief positive responses
    if any(word in feedback for word in ['good', 'great', 'thank you']) and len(feedback) < 50:
        return 'brief_positive'
    
    return 'other'

def analyze_feedback(df):
    """Perform the main analysis"""
    # Categorize all feedback
    df['category'] = df['feedback_clean'].apply(categorize_feedback)
    
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
    st.title("📊 CSAT Feedback Analysis Tool")
    st.markdown("### Medical Interpreter Service Feedback Analyzer")
    
    st.markdown("""
    This tool analyzes client feedback for medical interpreter services using keyword-based categorization 
    to identify common issues and positive trends. Upload your CSAT CSV file to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSAT CSV file", 
        type=['csv'],
        help="CSV should contain a 'ClientFeedback' column with client feedback text"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing feedback data..."):
            df = load_and_clean_data(uploaded_file)
            
        if df is not None:
            # Perform analysis
            df_analyzed = analyze_feedback(df)
            
            # Display results
            st.success(f"✅ Analyzed {len(df_analyzed)} feedback entries")
            
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
                file_name="csat_analysis_results.csv",
                mime="text/csv"
            )
    
    else:
        # Show methodology while waiting for upload
        st.subheader("🔧 Analysis Methodology")
        
        st.markdown("""
        **This tool uses the same approach as the manual analysis:**
        
        1. **Data Cleaning**: Removes empty feedback, trims whitespace
        2. **Keyword-Based Categorization**: Uses domain-specific dictionaries to identify issues
        3. **Hierarchical Classification**: Technical issues → Service issues → Positive feedback
        4. **Pattern Recognition**: Groups similar complaints and identifies trends
        5. **Actionable Insights**: Prioritizes findings by frequency and impact
        
        **Issue Categories Detected:**
        - 🔴 Connection Issues
        - 🔴 Disconnection Problems  
        - 🔴 Audio/Video Quality
        - 🔴 Professionalism Issues
        - 🔴 Accuracy/Speed Problems
        - 🔴 Availability Issues
        - 🟢 Excellent Service
        - 🟢 Professional Behavior
        - 🟢 Helpful/Patient Feedback
        """)

if __name__ == "__main__":
    main()
