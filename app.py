import os
import requests
import pandas as pd
import pypandoc
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fpdf import FPDF


# Load the .env file
load_dotenv()

# Access the environment variables
api_key = os.getenv("API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    api_key= api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


#Function for Scraping LinkedIn Data
def scrape_linkedin(user_name):
    url = "https://li-data-scraper.p.rapidapi.com/get-company-posts"

    querystring = {"username":user_name,"start":"0"}

    headers = {
        "x-rapidapi-key": "90e940a47fmsh7b1d1e49715aa23p1d6ab7jsnd8d29740fb6e",
        "x-rapidapi-host": "li-data-scraper.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    Linkedin_Data = response.json()

    return Linkedin_Data



#Function for scraping Company Website
def scrape_website(website_url):
    url = "https://scrapers-proxy2.p.rapidapi.com/parser"

    querystring = {"url":website_url,"auto_detect":"true"}

    headers = {
        "x-rapidapi-key": "90e940a47fmsh7b1d1e49715aa23p1d6ab7jsnd8d29740fb6e",
        "x-rapidapi-host": "scrapers-proxy2.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    Website_data = response.json()

    return Website_data 


# Function for Generating Company Analysis Report
def generate_company_report(LinkedInData, WebsiteData):
    """
    Generate a detailed company report based on LinkedIn profile and website analysis.
    
    Args:
    LinkedIn Company Profile Scraped Data: {LinkedInData}
    Official Company Website Scraped Data: {WebsiteData}
    
    Returns:
    - str: A detailed company report.
    """
    
    # Create a prompt template for generating the company report
    template = """
    Create a detailed company report for the following company details:
    
    LinkedIn Company Profile Scraped Data: {LinkedInData}

    
    Official Company Website Scraped Data: {WebsiteData}
    
    The report should be organized into the following sections:
    1. Executive Summary
    2. Company Overview
    3. Product or Service Offerings
    4. Market Positioning
    5. Team Overview
    6. Traction and Metrics
    7. Marketing and Branding
    8. Funding and Financial Overview
    9. Challenges and Risks
    10. Strategic Roadmap
    11. SWOT Analysis
    12. Recommendations
    
    Ensure the report is professional, concise, and insightful.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize LLM model
    llm = model  
    
    # Chain the prompt with the LLM and parse output
    report_generation_chain = prompt | llm | StrOutputParser()
    
    # Generate the report
    report = report_generation_chain.invoke({"LinkedInData":LinkedInData, 
                                             "WebsiteData":WebsiteData})
    
    return report



#Functions to Download Report
# pypandoc.download_pandoc()


# Function to create a docx
def create_docx(markdown_content, filename="overall_report.docx"):
    # Convert Markdown to docx
    try:
        pypandoc.convert_text(
            markdown_content,
            to='docx',
            format='md',
            outputfile=filename
        )

        st.success(f"Report successfully created: {filename}")
    except RuntimeError as e:
        st.error(f"Error in Report generation: {e}")



#Streamlit UI 
        
# Add a logo in the sidebar
st.sidebar.image("Logo.png", 
                 width=200)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Information", "Market Analysis", "Download Report"])


#First Page
if page == "User Information":

    st.title("VentureCatalyst")
    st.subheader('AI-Powered Success for Entrepreneurs.')
    st.header("Create Your Business Profile")

    with st.form("user_info_form"):
        business_name = st.text_input("Business Name")
        linkedin_user_name = st.text_input("LinkedIn user name")
        website_url = st.text_input("Official Website Url")
        
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            st.session_state["user_info"] = {
                "business_name": business_name,
                "linkedin_user_name": linkedin_user_name,
                "website_url": website_url
            }

            st.success("Information saved!")

            user_info = st.session_state["user_info"]

            #Scraping LinkedIn Data
            Linkedin_Data = scrape_linkedin(user_info["linkedin_user_name"])

            st.session_state["Linkedin_Data"] = Linkedin_Data

            st.success("LinkedIn Data extracted and saved!!!")

            #Scraping Website Data
            Website_Data = scrape_website(user_info["website_url"])

            st.session_state["Website_Data"] = Website_Data

            st.success("Website Data extracted and saved!!!")



#Second Page
if page == "Market Analysis":
    if "Linkedin_Data" not in st.session_state:
        st.warning("Please fill out the business form first.")
    
    else:
        Linkedin_Data = st.session_state["Linkedin_Data"]

        Website_Data = st.session_state["Website_Data"]

        st.title("Market Analysis Engine")
        
        # Simulate customer journey
        submit_button = st.button("Generate Report")
                
        if submit_button:
            Company_Report = generate_company_report(Linkedin_Data, Website_Data)

            st.session_state["Company_Analysis_Report"] = Company_Report

            st.subheader("Company profile Analysis Report:")
            st.write(Company_Report)



#Third Page
if page == "Download Report":
    #Download Report Docx
    if st.button("Download Report"):
        if "Company_Analysis_Report" in st.session_state:

            # Accessing the Report 
            Report_Generated = st.session_state["Company_Analysis_Report"] 

            user_info = st.session_state["user_info"]
            Brand_name = user_info["business_name"]

            # Generate Docx
            docx_filename = f"{Brand_name}_overall_report.docx"
            create_docx(Report_Generated, docx_filename)

            # Provide download link
            with open(docx_filename, "rb") as docx_file:
                st.download_button(
                    label= "Download Document",
                    data= docx_file,
                    file_name= docx_filename,
                    mime= "application/docx",
                )
        else:
            st.warning("Please Generate the Report in Market Analysis section first.")