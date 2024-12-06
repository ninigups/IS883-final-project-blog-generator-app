import os
import urllib.parse
from io import BytesIO
import time
import re

import openai
import streamlit as st
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

import pytesseract
from PIL import Image, ImageEnhance
import PyPDF2
import easyocr
import numpy as np

# Load API Keys
openai.api_key = st.secrets["TestKey1"]
os.environ["SERPER_API_KEY"] = st.secrets["SerperKey1"]

# Initialize the Google Serper API 
search = GoogleSerperAPIWrapper()
serper_tool = Tool(
    name="GoogleSerper",
    func=search.run,
    description="Useful for when you need to look up some information on the internet.",
)

# Custom CSS
st.set_page_config(
    page_title="Travel Planning Assistant",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body {
        background-color: #e3f2fd;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .st-expander {
        background-color: #f9f9f9;
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .st-expander-header {
        font-weight: bold;
        color: #2980b9;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 15px;
    }
    .stButton>button:hover {
        background-color: #1c598a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def display_card(title, content):
    return f"""
    <div style="background-color:#f9f9f9; padding:10px; border-radius:10px; margin-bottom:10px; border:1px solid #ddd;">
        <h4 style="color:#2980b9;">{title}</h4>
        <p>{content}</p>
    </div>
    """

def generate_maps_link(place_name, city_name):
    base_url = "https://www.google.com/maps/search/?api=1&query="
    full_query = f"{place_name}, {city_name}"
    return base_url + urllib.parse.quote(full_query)

def extract_place_name(activity_line):
    prefixes_to_remove = ["Visit", "Explore", "Rest", "the", "Last-minute Shopping in"]
    for prefix in prefixes_to_remove:
        if activity_line.lower().startswith(prefix.lower()):
            activity_line = activity_line.replace(prefix, "").strip()
    return activity_line

def format_flight_prices_with_chatgpt(raw_response, origin, destination, departure_date):
    try:
        prompt = f"""
        You are a helpful assistant. I received the following raw flight information:
        'Flights from {origin} to {destination} on {departure_date}':
        {raw_response}

        Please clean and reformat this information into a professional, readable format. Use bullet points,
        categories, or a table. Include the cheapest fare, airlines, and travel dates.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred while formatting the response: {e}"

def fetch_flight_prices(origin, destination, departure_date):
    try:
        query = f"flights from {origin} to {destination} on {departure_date}"
        raw_response = serper_tool.func(query)
        formatted_response = format_flight_prices_with_chatgpt(
            raw_response, origin, destination, departure_date
        )
        return formatted_response
    except Exception as e:
        return f"An error occurred while fetching or formatting flight prices: {e}"

def generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget):
    try:
        prompt_template = """
        You are a travel assistant. Create a detailed itinerary for a trip from {origin} to {destination}. 
        The user is interested in {interests}. The budget level is {budget}. 
        The travel dates are {travel_dates}. For each activity, include the expected expense in both local currency 
        and USD. Provide a total expense at the end. Include at least 5 places to visit as "Activity 1", "Activity 2", etc.
        """
        prompt = prompt_template.format(
            origin=origin,
            destination=destination,
            interests=", ".join(interests) if interests else "general activities",
            budget=budget,
            travel_dates=travel_dates
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred while generating the itinerary: {e}"

def create_pdf(itinerary, flight_prices):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    section_style = styles["Heading2"]
    text_style = styles["BodyText"]

    elements = []
    elements.append(Paragraph("Travel Itinerary", title_style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Itinerary:", section_style))
    for line in itinerary.splitlines():
        elements.append(Paragraph(line, text_style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Flight Prices:", section_style))
    for line in flight_prices.splitlines():
        elements.append(Paragraph(line, text_style))
    elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Pre-travel Section
st.title("🌍 Travel Planning Assistant")
st.write("Plan your perfect trip with personalized itineraries and flight suggestions!")

with st.sidebar:
    st.header("🛠️ Trip Details")
    origin = st.text_input("Flying From (Origin Airport/City)", placeholder="Enter your departure city/airport")
    destination = st.text_input("Flying To (Destination Airport/City)", placeholder="Enter your destination city/airport")
    travel_dates = st.date_input("📅 Travel Dates", [], help="Select your trip's start and end dates.")
    budget = st.selectbox("💰 Select your budget level", ["Low (up to $5,000)", "Medium ($5,000 to $10,000)", "High ($10,000+)"])
    interests = st.multiselect("🎯 Select your interests", ["Beach", "Hiking", "Museums", "Local Food", "Shopping", "Parks", "Cultural Sites", "Nightlife"])

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "flight_prices" not in st.session_state:
    st.session_state.flight_prices = None

if st.button("📝 Generate Travel Itinerary"):
    if not origin or not destination or len(travel_dates) != 2:
        st.error("⚠️ Please provide origin, destination, and a valid travel date range.")
    else:
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)
        with st.spinner("Fetching details..."):
            st.session_state.flight_prices = fetch_flight_prices(origin, destination, travel_dates[0].strftime("%Y-%m-%d"))
            st.session_state.itinerary = generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget)

if st.session_state.itinerary and st.session_state.flight_prices:
    st.success("✅ Your travel details are ready!")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(display_card("Itinerary", st.session_state.itinerary), unsafe_allow_html=True)
    with col2:
        st.markdown(display_card("Flight Prices", st.session_state.flight_prices), unsafe_allow_html=True)

    st.subheader("📍 Places to Visit with Map Links")
    activities = [
        line.split(":")[1].strip() 
        for line in st.session_state.itinerary.split("\n") 
        if ":" in line and "Activity" in line
    ]
    if activities:
        for activity in activities:
            place_name = extract_place_name(activity)
            if place_name:
                maps_link = generate_maps_link(place_name, destination)
                st.markdown(f"- **{place_name}**: [View on Google Maps]({maps_link})")
    else:
        st.write("No activities could be identified.")

    pdf_buffer = create_pdf(st.session_state.itinerary, st.session_state.flight_prices)
    st.download_button(
        label="📥 Download Itinerary as PDF",
        data=pdf_buffer,
        file_name="travel_itinerary.pdf",
        mime="application/pdf",
    )

# Post-Travel Review Section
st.header("Post-Travel Review")

# User input table for trip experience
st.subheader("Trip Experience Feedback")
parameters = [
    "Sight-seeing locations",
    "Hotels",
    "Food",
    "Local transport",
    "Local population (Friendliness, Helpfulness, Hospitable)",
    "Weather"
]
feedback_data = []
for param in parameters:
    rating = st.slider(f"{param} Rating (1-10)", 1, 10, 5, key=f"rating_{param}")
    review_text = st.text_input(f"Review for {param}", key=f"review_{param}", placeholder="Enter your review...")
    feedback_data.append({"Parameter": param, "Rating": rating, "Review": review_text})

if st.button("Submit Feedback"):
    feedback_df = pd.DataFrame(feedback_data)
    st.write("Your Trip Feedback:")
    st.write(feedback_df)

# Excel input for expenses
st.subheader("Upload Expenses (Excel File)")
expense_file = st.file_uploader("Upload an Excel file with expenses", type=["xlsx"], key="expense_file")
if expense_file is not None:
    expense_df = pd.read_excel(expense_file)
    st.write("Expenses from Excel:")
    st.write(expense_df)
else:
    expense_df = pd.DataFrame()

# 4. PDF (and image) input for expenses with OCR
st.subheader("Upload Receipts for Expense Extraction")
receipt_files = st.file_uploader("Upload receipts (PDF or Images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

reader = easyocr.Reader(['en'])

preprocess_contrast = st.slider("Increase Image Contrast Factor", 1.0, 3.0, 1.0, 0.1)
st.write("Use a higher factor if the receipt text is faint.")

improved_patterns = [
    r"(?i)(total\s*:?\s*\$?\s*([\d.,]+))",
    r"(?i)(grand\s*total\s*:?\s*\$?\s*([\d.,]+))",
    r"(?i)(amount\s*due\s*:?\s*\$?\s*([\d.,]+))",
    r"(?i)(tip\s*:?\s*\$?\s*([\d.,]+))"
]

def extract_amounts_from_text(text, patterns):
    amounts_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            val_str = m[-1].replace(",", "").strip()
            try:
                val = float(val_str)
                amounts_found.append(val)
            except:
                pass
    return amounts_found

def ocr_image(img, contrast_factor=1.0):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    # Use easyocr directly
    result = reader.readtext(np.array(img), detail=0)
    text = "\n".join(result)
    return text

extracted_expenses = []
text_content = ""  # store last processed text for classification

if receipt_files:
    for rfile in receipt_files:
        file_type = rfile.type
        text_content = ""
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(rfile)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
        else:
            # It's an image
            img = Image.open(rfile)
            text_content = ocr_image(img, contrast_factor=preprocess_contrast)

        st.write(f"**File:** {rfile.name}")
        st.write("**Extracted Text (OCR):**")
        st.text(text_content)
        amounts = extract_amounts_from_text(text_content, improved_patterns)
        if amounts:
            for amt in amounts:
                extracted_expenses.append({
                    "File": rfile.name,
                    "Extracted Amount": amt
                })
        else:
            st.write("No amounts found. Try adjusting contrast.")

if extracted_expenses:
    st.write("Extracted Expenses from Receipts:")
    extracted_df = pd.DataFrame(extracted_expenses)
    st.write(extracted_df)
else:
    extracted_df = pd.DataFrame()

# Consolidate OCR Receipts and Post-travel Receipts
if not expense_df.empty and not extracted_df.empty:
    combined_expenses = pd.concat([expense_df, extracted_df], ignore_index=True)
    st.subheader("Consolidated Expenses:")
    st.write(combined_expenses)
elif not expense_df.empty:
    st.subheader("Expenses (from Excel):")
    st.write(expense_df)
    combined_expenses = expense_df
elif not extracted_df.empty:
    st.subheader("Expenses (from Receipts):")
    st.write(extracted_df)
    combined_expenses = extracted_df
else:
    st.write("No expenses data available.")
    combined_expenses = pd.DataFrame()

# Tabular classification of output for expenses using an LLM agent
st.subheader("Classify Expenses into Categories")
# Categories: 'Food', 'Transport', 'Accommodation', 'Entertainment', 'Miscellaneous', 'Grand total'
if not combined_expenses.empty:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    tools = [serper_tool]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    # Take first expense text if we have OCR text:
    expense_text_for_class = text_content if text_content else "No OCR text"
    user_query = f"Classify this expense into one of ['Food','Transport','Accommodation','Entertainment','Miscellaneous','Grand total']. Expense details: {expense_text_for_class}"

    if st.button("Classify Expense"):
        with st.spinner("Classifying..."):
            agent_answer = agent.run(user_query)
        st.write("**Classification Result:**")
        st.write(agent_answer)
else:
    st.write("No expenses to classify.")

# 6. Generate Trip Experience Review button
st.subheader("Generate Trip Experience Review")
if "review_generated" not in st.session_state:
    st.session_state.review_generated = None

if st.button("Generate Review"):
    # Collect all feedback text
    all_reviews = " ".join([f"{row['Parameter']}: Rated {row['Rating']}, Review: {row['Review']}" for row in feedback_data])
    review_prompt = """
    SYSTEM: You are a travel assistant that summarizes user feedback.
    USER: Here are the user's feedback details:
    {}
    Please create a short review summarizing the user's trip experience in a friendly tone.
    """.format(all_reviews)
    review_chain_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    review_response = review_chain_llm(review_prompt)
    st.session_state.review_generated = review_response.content
    st.write("Generated Trip Experience Review:")
    st.write(st.session_state.review_generated)

# Classify the review generated based on parameters
if st.session_state.review_generated:
    st.subheader("Classify the Generated Review")
    sentiment_prompt = f"""
    SYSTEM: You are a sentiment analyst. Classify the user's review as 'Positive', 'Neutral', or 'Negative' based on the content.
    USER: {st.session_state.review_generated}
    """
    if st.button("Classify Generated Review"):
        sentiment_chain_llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)
        sentiment_response = sentiment_chain_llm(sentiment_prompt)
        st.write("Sentiment of the Review:")
        st.write(sentiment_response.content)
