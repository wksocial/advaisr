import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
from urllib.parse import urlencode, urlparse, urljoin
import json
import time
import os
import openai
from playwright.sync_api import sync_playwright
from fpdf import FPDF
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Marketing Advisor Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_keys_configured' not in st.session_state:
    st.session_state.api_keys_configured = False

# Sidebar for API configuration
with st.sidebar:
    st.title("🔑 API Configuration")
    
    with st.expander("Configure API Keys", expanded=not st.session_state.api_keys_configured):
        openai_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from platform.openai.com")
        google_api_key = st.text_input("Google API Key", type="password", help="Required for competitor analysis")
        google_cse_id = st.text_input("Google CSE ID", type="password", help="Custom Search Engine ID for Google")
        
        if st.button("Save API Keys"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                st.session_state.api_keys_configured = True
                st.success("API keys saved successfully!")
                st.rerun()
            else:
                st.error("OpenAI API Key is required")

# Only proceed if API keys are configured
if st.session_state.get('api_keys_configured', False):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {padding-top: 1rem;}
        .stTextInput input {padding: 10px !important;}
        .stButton button {width: 100%; padding: 10px !important;}
        .stExpander .stMarkdown {font-size: 16px !important;}
        .metric-card {border-radius: 10px; padding: 15px; background: #f0f2f6; margin-bottom: 15px;}
        .metric-title {font-size: 14px; color: #666;}
        .metric-value {font-size: 24px; font-weight: bold;}
        .tab-content {padding-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

    def ask_with_fusion(content, question="Summarize this website in one paragraph."):
        content = content[:10000] if len(content) > 10000 else content

        models = {
            "gpt-3.5-turbo": "GPT-3.5",
            "gpt-4": "GPT-4"
        }

        individual_answers = {}

        for model_name, label in models.items():
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are AI Advisor, a senior consultant who specializes in marketing, SEO strategy, and brand positioning."
                        },
                        {
                            "role": "user",
                            "content": f"Website content:\n\n{content}\n\nQuestion: {question}"
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                individual_answers[label] = response.choices[0].message.content
            except Exception as e:
                individual_answers[label] = f"Error from {label}: {e}"

        combined_answer_text = "\n\n".join([f"{label}:\n{ans}" for label, ans in individual_answers.items()])

        try:
            fusion_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are AI Advisor, an expert synthesizer who summarizes insights from multiple GPT models."
                    },
                    {
                        "role": "user",
                        "content": f"Here are multiple answers to the same question:\n\n{combined_answer_text}\n\nPlease synthesize the most accurate answer."
                    }
                ],
                temperature=0.5,
                max_tokens=600
            )
            return fusion_response.choices[0].message.content
        except Exception as e:
            return f"Fusion GPT error: {e}"

    def scrape_website(url):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                page.goto(url, timeout=30000)
                page.wait_for_selector('body', timeout=10000)
                html_content = page.content()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                metadata = {
                    'title': soup.title.string if soup.title else '',
                    'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else '',
                    'meta_keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else '',
                    'canonical': soup.find('link', attrs={'rel': 'canonical'})['href'] if soup.find('link', attrs={'rel': 'canonical'}) else '',
                }
                
                headings = {
                    'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
                    'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
                    'h3': [h.get_text(strip=True) for h in soup.find_all('h3')],
                }
                
                for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                    element.decompose()
                    
                main_content = ' '.join(soup.get_text(separator=' ', strip=True).split())
                
                base_domain = urlparse(url).netloc
                internal_links = []
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text(strip=True)
                    
                    if href.startswith('/'):
                        full_url = urljoin(url, href)
                        internal_links.append({'url': full_url, 'text': link_text})
                    elif base_domain in href:
                        internal_links.append({'url': href, 'text': link_text})
                
                browser.close()
                
                return {
                    'url': url,
                    'metadata': metadata,
                    'headings': headings,
                    'content': main_content,
                    'internal_links': internal_links
                }
                
            except Exception as e:
                browser.close()
                return {
                    'url': url,
                    'error': str(e)
                }

    def extract_keywords(content, n_gram_range=(1, 3), top_n=20):
        text = re.sub(r'[^\w\s]', ' ', content.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words='english')
        X = vectorizer.fit_transform([text])
        
        feature_names = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        
        keywords_df = pd.DataFrame({
            'keyword': feature_names,
            'count': counts
        })
        
        keywords_df = keywords_df.sort_values('count', ascending=False).head(top_n)
        
        return keywords_df

    def find_competitors(keyword, api_key, cse_id, num_results=5):
        base_url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            'key': api_key,
            'cx': cse_id,
            'q': keyword,
            'num': num_results
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            results = response.json().get('items', [])
            return [item['link'] for item in results]
        except Exception as e:
            return []

    def export_to_pdf(df, filename="seo_keywords.pdf", title="SEO Keyword Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        for idx, row in df.iterrows():
            keyword = row['keyword'].encode('latin-1', 'replace').decode('latin-1')
            score = int(row['count'])
            pdf.cell(0, 10, f"{idx+1}. {keyword} - Count: {score}", ln=True)

        pdf.output(filename)
        return filename

    def main():
        st.title("🚀 AI Marketing Advisor Pro")
        st.markdown("Analyze websites, get marketing insights, and discover SEO opportunities")
        
        # Tab layout
        tab1, tab2 = st.tabs(["Website Analysis", "Marketing Questions"])
        
        with tab1:
            st.subheader("Website Analysis Tool")
            user_input = st.text_input("Enter website URL:", placeholder="https://example.com")
            
            if user_input:
                if re.match(r"^(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input):
                    if not user_input.startswith(("http://", "https://")):
                        user_input = "https://" + user_input
                    
                    with st.spinner("🔍 Analyzing website..."):
                        site_data = scrape_website(user_input)
                        
                        if 'error' in site_data:
                            st.error(f"❌ Error scraping website: {site_data['error']}")
                        else:
                            # Metrics row
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown('<div class="metric-card"><div class="metric-title">Title Length</div><div class="metric-value">{}</div></div>'.format(
                                    len(site_data['metadata']['title'])), unsafe_allow_html=True)
                            with col2:
                                st.markdown('<div class="metric-card"><div class="metric-title">Meta Description Length</div><div class="metric-value">{}</div></div>'.format(
                                    len(site_data['metadata']['meta_description'])), unsafe_allow_html=True)
                            with col3:
                                st.markdown('<div class="metric-card"><div class="metric-title">H1 Count</div><div class="metric-value">{}</div></div>'.format(
                                    len(site_data['headings']['h1'])), unsafe_allow_html=True)
                            
                            # Main content columns
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                with st.container(border=True):
                                    st.markdown("### 📌 Basic Info")
                                    st.write(f"**Title:** {site_data['metadata']['title']}")
                                    st.write(f"**Meta Description:** {site_data['metadata']['meta_description']}")
                                    st.write(f"**Canonical URL:** {site_data['metadata']['canonical']}")
                                
                                with st.container(border=True):
                                    st.markdown("### 🏷️ Headings")
                                    st.write("**H1:**")
                                    for h1 in site_data['headings']['h1'][:5]:
                                        st.write(f"- {h1}")
                                    
                                    st.write("**H2:**")
                                    for h2 in site_data['headings']['h2'][:5]:
                                        st.write(f"- {h2}")
                            
                            with col2:
                                with st.container(border=True):
                                    st.markdown("### 🔍 Content Analysis")
                                    analysis = ask_with_fusion(site_data['content'])
                                    st.write(analysis)
                                
                                with st.container(border=True):
                                    st.markdown("### 🔑 Top Keywords")
                                    keywords = extract_keywords(site_data['content'])
                                    st.dataframe(keywords, use_container_width=True)
                                    
                                    # Export buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("📤 Export Keywords to CSV"):
                                            csv = keywords.to_csv(index=False)
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name="seo_keywords.csv",
                                                mime="text/csv"
                                            )
                                    with col2:
                                        if st.button("📤 Export to PDF"):
                                            pdf_file = export_to_pdf(keywords)
                                            with open(pdf_file, "rb") as f:
                                                st.download_button(
                                                    label="Download PDF",
                                                    data=f,
                                                    file_name="seo_keywords.pdf",
                                                    mime="application/pdf"
                                                )
                            
                            # Competitor analysis section
                            if st.button("🔎 Find Competitors", use_container_width=True):
                                if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                                    competitors = find_competitors(
                                        site_data['metadata']['title'] or user_input,
                                        GOOGLE_API_KEY,
                                        GOOGLE_CSE_ID
                                    )
                                    
                                    if competitors:
                                        with st.container(border=True):
                                            st.markdown("### 🏆 Top Competitors")
                                            for i, comp in enumerate(competitors, 1):
                                                st.write(f"{i}. [{comp}]({comp})")
                                    else:
                                        st.warning("No competitors found or there was an error with the Google API")
                                else:
                                    st.error("Google API keys are required for competitor analysis")
                else:
                    st.error("Please enter a valid website URL")
        
        with tab2:
            st.subheader("Marketing Question Answering")
            question = st.text_area("Ask your marketing question:", 
                                  placeholder="How can I improve my website's conversion rate?",
                                  height=150)
            
            if st.button("Get Answer", key="question_btn", use_container_width=True):
                if question:
                    with st.spinner("🤖 Generating answer..."):
                        analysis = ask_with_fusion(question)
                        st.markdown("### 💡 AI Advisor Response")
                        st.write(analysis)
                else:
                    st.warning("Please enter a question")

    if __name__ == "__main__":
        main()
else:
    st.warning("Please configure your API keys in the sidebar to use the application")
