#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# 🧠 SEO Keyword Analyzer (Modular App)
# This notebook allows a user to input any website URL and optional keyword.
# It will:
# - Scrape SEO metadata, headings, content, and internal links
# - Extract and clean relevant keyword phrases
# - Use Google CSE API to find competitors
# - Compare keywords with competitors
# - Export results as CSV and JSON

# Run this command to upgrade the OpenAI package

import subprocess

subprocess.run(["pip", "install", "--upgrade", "openai"])


# Let's also check the version after upgrading
import openai
print(f"OpenAI package version after upgrade: {openai.__version__}")

# If you want to see all the details about the installed package
#pip install pip show openai
pip install -r requirements.txt

# In[ ]:
pip install openai


# In[ ]:


pip install fpdf


# In[ ]:
# 📦 Install both Selenium and Playwright
pip install selenium playwright



# In[ ]:


# 🧠 Install Chromium for Playwright (only needs to be run once)
pip install playwright


# In[ ]:


import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
from urllib.parse import urlencode, urlparse
import json
import time
import os
import openai
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Add the ask_with_fusion() Function
"You are AI Advisor, a senior consultant who specializes in marketing, SEO strategy, and brand positioning. You respond like a human advisor — direct, smart, and focused on impact."
# ✅ Multi-model GPT fusion function
def ask_with_fusion(content, question="Summarize this website in one paragraph."):
    content = content[:10000] if len(content) > 10000 else content

    models = {
        "gpt-3.5-turbo": "GPT-3.5",
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "Backup GPT-3.5"  # Simulated 3rd model
    }

    individual_answers = {}

    for model_name, label in models.items():
        print(f"🔄 Asking {label}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are AI Advisor, a senior consultant who specializes in marketing, SEO strategy, and brand positioning. You respond like a human advisor — direct, smart, and focused on impact."
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
            individual_answers[label] = f"❌ Error from {label}: {e}"

    # Combine answers
    combined_answer_text = "\n\n".join([f"{label}:\n{ans}" for label, ans in individual_answers.items()])

    print("🧠 Synthesizing final answer with GPT-4...")

    try:
        fusion_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are AI Advisor, an expert synthesizer who summarizes insights from multiple GPT models for marketing and strategic clarity."
                },
                {
                    "role": "user",
                    "content": f"Here are multiple answers to the same question:\n\n{combined_answer_text}\n\nPlease synthesize the most accurate, insightful, and complete answer possible."
                }
            ],
            temperature=0.5,
            max_tokens=600
        )
        return fusion_response.choices[0].message.content
    except Exception as e:
        return f"❌ Fusion GPT error: {e}"


# In[ ]:


#Get User Input
#Handle URL Input Cleanly
# ✅ Updated: Handle plain domain input (e.g. "meals4heels.com")

import re

# ✅ One smart input for users to speak to the AI Advisor
user_input = input("🧠 Ask the AI Advisor: enter a website URL, business, or a marketing-related question: ").strip()

# Detect if it's a URL
def is_url(text):
    return bool(re.match(r"^(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text))

# Fix scheme if missing
if is_url(user_input) and not user_input.startswith(("http://", "https://")):
    user_input = "https://" + user_input

input_type = "url" if is_url(user_input) else "text"
print(f"📥 Input type detected: {input_type}")





# In[ ]:


#Then Handle It Like This
# Handle URL input
# ✅ Unified Logic for URL or Free Text
if input_type == "url":
    # Scrape the website
    site_data = scrape_site_all_sources(user_input)
    content = site_data["content"]
    print("🔍 Site scraped. Running AI Advisor analysis...")

    # Run first GPT analysis
    question = "Summarize what this website is about and provide initial marketing insights."
    final_answer = ask_with_fusion(content, question)
    print("\n🧠 AI Advisor Summary:\n")
    print(final_answer)
else:
    # User input is a direct question or phrase
    content = user_input
    print("🧠 Running AI Advisor analysis...")
    final_answer = ask_with_fusion(content, content)
    print("\n🧠 AI Advisor Response:\n")
    print(final_answer)

# ✅ Follow-up loop
while True:
    print("\n⚡ What would you like the power to do next?")
    print("Choose an option or ask a new question:")

    if input_type == "url":
        options = {
            "1": "Audit SEO and accessibility",
            "2": "Analyze keywords and content strategy",
            "3": "Find top competitors",
            "4": "Generate meta title & description"
        }
    else:
        options = {
            "1": "Suggest content angles",
            "2": "Generate SEO keywords",
            "3": "Recommend marketing channels",
            "4": "Brainstorm CTA ideas"
        }

    for key, label in options.items():
        print(f"{key}️⃣ {label}")

    followup = input("\n🧠 Ask anything or enter a number (or press Enter to exit): ").strip()

    if not followup:
        print("✅ Done. No more follow-ups.")
        break

    # If it's a numbered follow-up, use the predefined prompt
    if followup in options:
        followup_question = options[followup]
    else:
        followup_question = followup  # Treat as freeform GPT prompt

    print(f"\n🔁 Running: {followup_question}")
    followup_answer = ask_with_fusion(content, followup_question)
    print("\n📊 Follow-up Analysis:\n")
    print(followup_answer)



# In[ ]:


#Scrape the Target Website
import time
import shutil
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from playwright.sync_api import sync_playwright
import requests

def get_soup_requests(url):
    try:
        res = requests.get(url, timeout=10)
        return BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"❌ Error in get_soup_requests: {e}")
        return None

def get_soup_selenium(url):
    try:
        chrome_path = shutil.which("chromedriver")
        if not chrome_path:
            raise EnvironmentError("ChromeDriver not found")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(service=Service(chrome_path), options=options)
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        return BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"❌ Error in get_soup_selenium: {e}")
        return None

def get_soup_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")
            html = page.content()
            browser.close()
        return BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"❌ Error in get_soup_playwright: {e}")
        return None

def parse_soup(soup, domain):
    data = {
        "title": soup.title.string if soup.title else None,
        "meta_description": None,
        "headings": {"h1": [], "h2": [], "h3": []},
        "content": "",
        "internal_links": []
    }
    desc = soup.find("meta", attrs={"name": "description"})
    data["meta_description"] = desc["content"] if desc and "content" in desc.attrs else None

    for h in ["h1", "h2", "h3"]:
        data["headings"][h] = [tag.get_text(strip=True) for tag in soup.find_all(h)]

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    data["content"] = soup.get_text(separator=' ', strip=True)

    internal_links = [a["href"] for a in soup.find_all("a", href=True)
                      if domain in a["href"] or a["href"].startswith("/")]
    data["internal_links"] = list(set(internal_links))
    return data

def scrape_site_all_sources(url):
    domain = urlparse(url).netloc
    soup1 = get_soup_requests(url)
    soup2 = get_soup_selenium(url)
    soup3 = get_soup_playwright(url)

    parsed_data = {
        "url": url,
        "title": None,
        "meta_description": None,
        "headings": {"h1": [], "h2": [], "h3": []},
        "content": "",
        "internal_links": []
    }

    soups = [soup for soup in [soup1, soup2, soup3] if soup]

    for soup in soups:
        parsed = parse_soup(soup, domain)

        if not parsed_data["title"] and parsed["title"]:
            parsed_data["title"] = parsed["title"]
        if not parsed_data["meta_description"] and parsed["meta_description"]:
            parsed_data["meta_description"] = parsed["meta_description"]

        parsed_data["content"] += " " + parsed["content"]
        for h in ["h1", "h2", "h3"]:
            parsed_data["headings"][h].extend(parsed["headings"][h])
        parsed_data["internal_links"].extend(parsed["internal_links"])

    for h in ["h1", "h2", "h3"]:
        parsed_data["headings"][h] = list(set(parsed_data["headings"][h]))
    parsed_data["internal_links"] = list(set(parsed_data["internal_links"]))
    parsed_data["content"] = parsed_data["content"].strip()

    return parsed_data


# In[ ]:


#Keyword Extraction with Cleaning
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# ✅ Keyword Extraction with Cleaning
def extract_keywords(text, relevance_terms=None, top_n=10):
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english').fit([text])
    ngrams = vectorizer.transform([text])
    sum_words = ngrams.sum(axis=0)
    keywords = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    # 🔍 Clean and filter
    blacklist = ["st", "ave", "blvd", "drive", "comments", "reddit"]
    filtered = [
        (word, score) for word, score in keywords
        if not any(char.isdigit() for char in word)
        and not any(bl in word.lower() for bl in blacklist)
    ]

    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(filtered, columns=["keyword", "score"])

    # 🎯 Optional relevance filtering
    if relevance_terms:
        match = df[df["keyword"].str.contains('|'.join(relevance_terms), case=False)]
        if len(match) < top_n:
            extra = df[~df["keyword"].isin(match["keyword"])].head(top_n - len(match))
            df = pd.concat([match, extra])
        else:
            df = match.head(top_n)
    else:
        df = df.head(top_n)

    return df.reset_index(drop=True)


# In[ ]:


#Get Google Competitor URLs
from urllib.parse import urlencode
import requests

# ✅ Get Google Competitor URLs via CSE
def get_google_competitors(query, max_results=5):
    search_url = f"https://www.googleapis.com/customsearch/v1?{urlencode({'q': query, 'key': GOOGLE_API_KEY, 'cx': GOOGLE_CSE_ID})}"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()  # Catch HTTP errors
        results = response.json().get("items", [])
        return [item["link"] for item in results[:max_results]]
    
    except Exception as e:
        print(f"❌ Google CSE error: {e}")
        return []


# In[ ]:


# ✅ Analyze Target Site (after scraping)
site_data = scrape_site_all_sources(user_url)

print("🧠 Site Analysis:")
print("🔹 Title:", site_data.get("title", "N/A"))
print("🔹 Meta Description:", site_data.get("meta_description", "N/A"))

print("\n🔹 Headings Found:")
for level, tags in site_data.get("headings", {}).items():
    print(f"  {level.upper()}: {tags[:3]}{'...' if len(tags) > 3 else ''}")

print("\n🔹 Internal Links (sample):")
for link in site_data.get("internal_links", [])[:5]:
    print("  -", link)



# In[ ]:


# ✅ Extract Keywords from Target Site (with content check)

relevance_terms = user_keyword.lower().split() if 'user_keyword' in locals() and user_keyword else None

# Only run if site content is long enough
if site_data.get("content") and len(site_data["content"].split()) > 5:
    print("🔍 Extracting SEO keywords from site content...")
    site_keywords = extract_keywords(site_data["content"], relevance_terms=relevance_terms)
    display(site_keywords)
else:
    site_keywords = pd.DataFrame(columns=["keyword", "score"])
    print("⚠️ Not enough content to extract keywords from target site.")



# In[ ]:


# ✅ Analyze Competitor Sites
print("🔍 Finding competitor URLs...")
competitor_urls = get_google_competitors(user_keyword or site_data.get("title", ""))
print("🌐 Competitor URLs found:", len(competitor_urls))

all_keywords = []

for url in competitor_urls:
    print(f"🕸 Scraping: {url}")
    try:
        content = scrape_site_all_sources(url).get("content", "")
        if content and len(content.split()) > 5:
            phrases = extract_keywords(content, relevance_terms=relevance_terms)
            phrases["source_url"] = url
            all_keywords.append(phrases)
        else:
            print("⚠️ Skipped (not enough content)")
    except Exception as e:
        print(f"❌ Error scraping {url}:", e)
    time.sleep(2)  # Be polite to servers

# Combine and sort
if all_keywords:
    combined_df = pd.concat(all_keywords, ignore_index=True)
    combined_df = combined_df.sort_values(by="score", ascending=False)
    display(combined_df.head(10))
else:
    print("⚠️ No keyword data extracted from competitors.")



# In[ ]:


import re
import requests
from urllib.parse import urlencode
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bs4 import BeautifulSoup

# 🔍 Get URLs from Google Custom Search
def get_urls_google(keyword, max_results=5):
    search_url = f"https://www.googleapis.com/customsearch/v1?{urlencode({'q': keyword, 'key': GOOGLE_API_KEY, 'cx': GOOGLE_CSE_ID})}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        results = response.json().get("items", [])
        return [item["link"] for item in results[:max_results]]
    except Exception as e:
        print(f"❌ Google CSE error: {e}")
        return []

# 🔍 Get URLs from SerpAPI
def get_urls_serpapi(keyword, max_results=5):
    serp_url = f"https://serpapi.com/search?{urlencode({'q': keyword, 'api_key': SERPAPI_KEY, 'engine': 'google'})}"
    try:
        response = requests.get(serp_url)
        response.raise_for_status()
        items = response.json().get("organic_results", [])
        return [item["link"] for item in items[:max_results]]
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return []

# 🌐 Lightweight page scraper
def scrape_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=' ')
        return re.sub(r'\s+', ' ', text).lower().strip()
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# 🧠 Phrase extraction (bigrams/trigrams + filters)
def extract_phrases(text, relevance_terms=None, top_n=10):
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english').fit([text])
    ngrams = vectorizer.transform([text])
    sum_words = ngrams.sum(axis=0)
    keywords = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    # ✂️ Filter out numbers, addresses, junk
    blacklist_terms = ["st", "ave", "blvd", "street", "drive", "rd", "comments", "reddit"]
    cleaned_keywords = [
        (word, score) for word, score in keywords
        if not any(char.isdigit() for char in word)
        and not any(bl in word.lower() for bl in blacklist_terms)
    ]

    # 📊 Convert to DataFrame
    df = pd.DataFrame(sorted(cleaned_keywords, key=lambda x: x[1], reverse=True), columns=["keyword", "score"])

    # 🎯 Filter by relevance if provided
    if relevance_terms:
        filtered = df[df["keyword"].str.contains('|'.join(relevance_terms), case=False)]
        if len(filtered) < top_n:
            extra = df[~df["keyword"].isin(filtered["keyword"])].head(top_n - len(filtered))
            df = pd.concat([filtered, extra], ignore_index=True)
        else:
            df = filtered.head(top_n)
    else:
        df = df.head(top_n)

    return df.reset_index(drop=True)



# In[ ]:


#Export to CSV & JSON & pdf
import json
from datetime import datetime
from fpdf import FPDF

# 📄 PDF Export Helper (safe for special characters)
def export_to_pdf(df, filename="seo_keywords.pdf", title="SEO Keyword Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    for idx, row in df.iterrows():
        keyword = row['keyword'].encode('latin-1', 'replace').decode('latin-1')  # Safe encoding
        score = int(row['score'])
        pdf.cell(0, 10, f"{idx+1}. {keyword} - Score: {score}", ln=True)

    pdf.output(filename)
    print(f"✅ Exported to {filename}")

# ✅ Prompt user for export type
if not combined_df.empty:
    print("\n💾 How would you like to export the keyword results?")
    print("1️⃣ CSV")
    print("2️⃣ JSON")
    print("3️⃣ PDF")
    print("4️⃣ All of the above")
    print("⏭️ Press Enter to skip")

    export_choice = input("📤 Enter your choice (1, 2, 3, 4 or Enter): ").strip()

    # Add timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_filename = f"seo_keywords_{timestamp}"

    if export_choice == "1":
        combined_df.to_csv(f"{base_filename}.csv", index=False)
        print(f"✅ Exported to {base_filename}.csv")

    elif export_choice == "2":
        with open(f"{base_filename}.json", "w") as f:
            json.dump(combined_df.to_dict(orient="records"), f, indent=2)
        print(f"✅ Exported to {base_filename}.json")

    elif export_choice == "3":
        export_to_pdf(combined_df, f"{base_filename}.pdf")

    elif export_choice == "4":
        combined_df.to_csv(f"{base_filename}.csv", index=False)
        with open(f"{base_filename}.json", "w") as f:
            json.dump(combined_df.to_dict(orient="records"), f, indent=2)
        export_to_pdf(combined_df, f"{base_filename}.pdf")
        print(f"✅ Exported to CSV, JSON, and PDF")

    else:
        print("⏭️ Export skipped.")

else:
    print("⚠️ No keyword data available to export.")



# In[1]:


import os

# Force define path
file_path = "requirements.txt"

# Safety: remove if exists
if os.path.exists(file_path):
    os.remove(file_path)
    print("⚠️ Removed old file.")

# Write a new requirements.txt
try:
    with open(file_path, "w") as f:
        f.write("""streamlit
openai
beautifulsoup4
requests
fpdf
pandas
scikit-learn
selenium
playwright
""")
    print("✅ File created.")
except Exception as e:
    print("❌ Failed to create file:", e)

# Check if file was created and readable
if os.path.exists(file_path):
    print("📂 File exists. Preview:")
    with open(file_path, "r") as f:
        print(f.read())
else:
    print("❌ Still missing after creation attempt.")


# In[ ]:





# In[19]:


# Option 1: Use subprocess to run pip as a command
import subprocess
subprocess.check_call(['pip', 'install', '--upgrade', 'openai'])

# Option 2: Use sys.executable to ensure the correct Python environment
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'openai'])

# Option 3: Use os.system (less recommended but simpler)
import os
os.system('pip install --upgrade openai')


# In[21]:


# Option 1: Use subprocess to run pip as a command
import subprocess
subprocess.check_call(['pip', 'install', '--upgrade', 'openai'])

# Option 2: Use sys.executable to ensure the correct Python environment
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'openai'])

# Option 3: Use os.system (less recommended but simpler)
import os
os.system('pip install --upgrade openai')


# In[23]:


# Option 1: Use subprocess to run pip as a command
import subprocess
subprocess.check_call(['pip', 'install', '--upgrade', 'openai'])

# Option 2: Use sys.executable to ensure the correct Python environment
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'openai'])

# Option 3: Use os.system (less recommended but simpler)
import os
os.system('pip install --upgrade openai')


# In[25]:


# Option 1: Use subprocess to run pip as a command
import subprocess
subprocess.check_call(['pip', 'install', '--upgrade', 'openai'])

# Option 2: Use sys.executable to ensure the correct Python environment
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'openai'])

# Option 3: Use os.system (less recommended but simpler)
import os
os.system('pip install --upgrade openai')


# In[27]:


try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # For example
else:
    # Regular Python alternative
    print(df)  # For example


# # Assistant
# The error occurs because the code is trying to display or print a DataFrame called `df`, but this DataFrame has not been defined anywhere in your code.
# 
# Would you like me to provide the corrected code?

# In[37]:


import pandas as pd  # Import pandas library

# Create a sample DataFrame since 'df' is not defined
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # Now df is defined
else:
    # Regular Python alternative
    print(df)  # Now df is defined


# In[39]:


import pandas as pd  # Import pandas library

# Create a sample DataFrame since 'df' is not defined
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # Now df is defined
else:
    # Regular Python alternative
    print(df)  # Now df is defined


# In[41]:


import pandas as pd  # Import pandas library

# Create a sample DataFrame since 'df' is not defined
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # Now df is defined
else:
    # Regular Python alternative
    print(df)  # Now df is defined


# In[43]:


import pandas as pd  # Import pandas library

# Create a sample DataFrame since 'df' is not defined
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # Now df is defined
else:
    # Regular Python alternative
    print(df)  # Now df is defined


# In[45]:


import pandas as pd  # Import pandas library

# Create a sample DataFrame since 'df' is not defined
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

try:
    # Check if running in IPython
    get_ipython()
    is_ipython = True
except NameError:
    is_ipython = False
    
# Then use conditional logic based on the environment
if is_ipython:
    # IPython-specific code
    display(df)  # Now df is defined
else:
    # Regular Python alternative
    print(df)  # Now df is defined


# In[47]:


try:
    # Your code here that might raise an exception
    some_function()
    some_variable = some_calculation()
except SomeException:
    # Handle the exception
    handle_error()


# In[49]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[50]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[51]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[52]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[54]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[55]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[56]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[58]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[59]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[60]:


# First, let's check the current version
import openai
print(f"Current OpenAI version: {openai.__version__}")

# Force reinstall with pip (sometimes needed to resolve dependency issues)
get_ipython().system('pip install --upgrade --force-reinstall openai')

# Check if we can use the older API style
try:
    # If this works, we're using the older version
    openai.api_key = "your-api-key-here"  # Replace with your actual API key
    print("Using older OpenAI API style (pre-1.0)")
    
    # Example of how to use the older API style
    def get_completion(prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    
    # For chat models in older API style
    def get_chat_completion(messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message["content"]
    
    print("OpenAI functions loaded successfully with older API style")
    
except Exception as e:
    print(f"Error with older API style: {e}")
    
# Check the version again after reinstall
import importlib
importlib.reload(openai)
print(f"OpenAI version after reinstall: {openai.__version__}")

# Print the location of the openai package
import inspect
print(f"OpenAI package location: {inspect.getfile(openai)}")

# List all available attributes in the openai module
print("\nAvailable attributes in openai module:")
for attr in dir(openai):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")


# In[62]:


# Import the os module
import os

# Now you can use os functions
# Here are some common examples:

# Get current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# List files in the current directory
files = os.listdir()
print(f"Files in current directory: {files[:5]}")  # Show first 5 files

# Check if a file or directory exists
file_exists = os.path.exists('example.txt')
print(f"Does 'example.txt' exist? {file_exists}")

# Create a directory
try:
    os.makedirs('new_folder', exist_ok=True)
    print("Directory 'new_folder' created or already exists")
except Exception as e:
    print(f"Error creating directory: {e}")

# Get environment variables
path_var = os.environ.get('PATH')
print(f"PATH environment variable: {path_var[:100]}...")  # Show first 100 chars

# Join paths in an OS-independent way
joined_path = os.path.join(current_dir, 'data', 'file.csv')
print(f"Joined path: {joined_path}")

# Get file information
if os.path.exists('example.txt'):
    file_size = os.path.getsize('example.txt')
    print(f"File size: {file_size} bytes")
    
    file_mod_time = os.path.getmtime('example.txt')
    import datetime
    print(f"Last modified: {datetime.datetime.fromtimestamp(file_mod_time)}")


# In[63]:


# Import the os module
import os

# Now you can use os functions
# Here are some common examples:

# Get current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# List files in the current directory
files = os.listdir()
print(f"Files in current directory: {files[:5]}")  # Show first 5 files

# Check if a file or directory exists
file_exists = os.path.exists('example.txt')
print(f"Does 'example.txt' exist? {file_exists}")

# Create a directory
try:
    os.makedirs('new_folder', exist_ok=True)
    print("Directory 'new_folder' created or already exists")
except Exception as e:
    print(f"Error creating directory: {e}")

# Get environment variables
path_var = os.environ.get('PATH')
print(f"PATH environment variable: {path_var[:100]}...")  # Show first 100 chars

# Join paths in an OS-independent way
joined_path = os.path.join(current_dir, 'data', 'file.csv')
print(f"Joined path: {joined_path}")

# Get file information
if os.path.exists('example.txt'):
    file_size = os.path.getsize('example.txt')
    print(f"File size: {file_size} bytes")
    
    file_mod_time = os.path.getmtime('example.txt')
    import datetime
    print(f"Last modified: {datetime.datetime.fromtimestamp(file_mod_time)}")


# In[ ]:





# In[67]:


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
import numpy as np
from fpdf import FPDF
from datetime import datetime

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Multi-model GPT fusion function
def ask_with_fusion(content, question="Summarize this website in one paragraph."):
    content = content[:10000] if len(content) > 10000 else content

    models = {
        "gpt-3.5-turbo": "GPT-3.5",
        "gpt-4": "GPT-4",
        "gpt-3.5-turbo": "Backup GPT-3.5"  # Simulated 3rd model
    }

    individual_answers = {}

    for model_name, label in models.items():
        print(f"🔄 Asking {label}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are AI Advisor, a senior consultant who specializes in marketing, SEO strategy, and brand positioning. You respond like a human advisor — direct, smart, and focused on impact."
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
            individual_answers[label] = f"❌ Error from {label}: {e}"

    # Combine answers
    combined_answer_text = "\n\n".join([f"{label}:\n{ans}" for label, ans in individual_answers.items()])

    print("🧠 Synthesizing final answer with GPT-4...")

    try:
        fusion_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are AI Advisor, an expert synthesizer who summarizes insights from multiple GPT models for marketing and strategic clarity."
                },
                {
                    "role": "user",
                    "content": f"Here are multiple answers to the same question:\n\n{combined_answer_text}\n\nPlease synthesize the most accurate, insightful, and complete answer possible."
                }
            ],
            temperature=0.5,
            max_tokens=600
        )
        return fusion_response.choices[0].message.content
    except Exception as e:
        return f"❌ Fusion GPT error: {e}"

# Function to scrape website content using Playwright
def scrape_website(url):
    print(f"🔍 Scraping {url}...")
    
    # Ensure URL has proper scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, timeout=30000)
            # Wait for content to load
            page.wait_for_selector('body', timeout=10000)
            
            # Get the full HTML content
            html_content = page.content()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = {
                'title': soup.title.string if soup.title else '',
                'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else '',
                'meta_keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else '',
                'canonical': soup.find('link', attrs={'rel': 'canonical'})['href'] if soup.find('link', attrs={'rel': 'canonical'}) else '',
            }
            
            # Extract headings
            headings = {
                'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
                'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
                'h3': [h.get_text(strip=True) for h in soup.find_all('h3')],
            }
            
            # Extract main content
            # Remove script, style elements and comments
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
                
            main_content = ' '.join(soup.get_text(separator=' ', strip=True).split())
            
            # Extract internal links
            base_domain = urlparse(url).netloc
            internal_links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text(strip=True)
                
                # Handle relative URLs
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
            print(f"❌ Error scraping {url}: {e}")
            return {
                'url': url,
                'error': str(e)
            }

# Function to extract keywords from content
def extract_keywords(content, n_gram_range=(1, 3), top_n=20):
    print("🔑 Extracting keywords...")
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', content.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract n-grams
    vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words='english')
    X = vectorizer.fit_transform([text])
    
    # Get all features
    feature_names = vectorizer.get_feature_names_out()
    
    # Get counts for each n-gram
    counts = X.toarray()[0]
    
    # Create DataFrame with features and counts
    keywords_df = pd.DataFrame({
        'keyword': feature_names,
        'count': counts
    })
    
    # Sort by count and get top N
    keywords_df = keywords_df.sort_values('count', ascending=False).head(top_n)
    
    return keywords_df

# Function to find competitors using Google CSE API
def find_competitors(keyword, api_key, cse_id, num_results=5):
    print(f"🔍 Finding competitors for '{keyword}'...")
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': keyword,
        'num': num_results
    }
    
   
        


# In[ ]:




