import os
import urllib.parse
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import openai
import streamlit as st
import time
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance
import PyPDF2
import re
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Load API keys
os.environ["OPENAI_API_KEY"] = st.secrets['TestKey1']
os.environ["SERPER_API_KEY"] = st.secrets["SerperKey1"]

# Function to generate Google Maps link
def generate_maps_link(place_name, city_name):
    base_url = "https://www.google.com/maps/search/?api=1&query="
    full_query = f"{place_name}, {city_name}"
    return base_url + urllib.parse.quote(full_query)

# Function to clean and extract valid place names
def extract_place_name(activity_line):
    prefixes_to_remove = ["Visit", "Explore", "Rest", "the", "Last-minute Shopping in"]
    for prefix in prefixes_to_remove:
        if activity_line.lower().startswith(prefix.lower()):
            activity_line = activity_line.replace(prefix, "").strip()
    return activity_line

# Initialize the Google Serper API Wrapper
search = GoogleSerperAPIWrapper()
serper_tool = Tool(
    name="GoogleSerper",
    func=search.run,
    description="Useful for when you need to look up some information on the internet.",
)

# Function to query ChatGPT for better formatting
def format_flight_prices_with_chatgpt(raw_response, origin, destination, departure_date):
    try:
        prompt = f"""
        You are a helpful assistant. I received the following raw flight information for a query:
        'Flights from {origin} to {destination} on {departure_date}':
        {raw_response}

        Please clean and reformat this information into a professional, readable format. Use bullet points,
        categories, or a table wherever appropriate to make it easy to understand. Also include key highlights
        like the cheapest fare, airlines, and travel dates. Ensure that any missing or irrelevant text is ignored.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred while formatting the response: {e}"

# Function to fetch flight prices and format them with ChatGPT
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

# Function to generate a detailed itinerary using ChatGPT
def generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget):
    try:
        prompt_template = """
        You are a travel assistant. Create a detailed itinerary for a trip from {origin} to {destination}. 
        The user is interested in {interests}. The budget level is {budget}. 
        The travel dates are {travel_dates}. For each activity, include the expected expense in both local currency 
        and USD. Provide a total expense at the end. Include at least 5 places to visit and list them as "Activity 1", "Activity 2", etc.
        """
        prompt = prompt_template.format(
            origin=origin,
            destination=destination,
            interests=", ".join(interests) if interests else "general activities",
            budget=budget,
            travel_dates=travel_dates
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred while generating the itinerary: {e}"

# Function to create a PDF from itinerary and flight prices
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

# Initialize session state variables
if "post_trip_active" not in st.session_state:
    st.session_state.post_trip_active = False
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "flight_prices" not in st.session_state:
    st.session_state.flight_prices = None

# Streamlit UI configuration
st.set_page_config(
    page_title="Travel Planning Assistant",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
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

# Function to display content in cards
def display_card(title, content):
    return f"""
    <div style="background-color:#f9f9f9; padding:10px; border-radius:10px; margin-bottom:10px; border:1px solid #ddd;">
        <h4 style="color:#2980b9;">{title}</h4>
        <p>{content}</p>
    </div>
    """

# App Title
st.title("🌍 Travel Planning Assistant")
st.write("Plan your perfect trip with personalized itineraries and flight suggestions!")

# Sidebar Inputs
with st.sidebar:
    st.header("🛠️ Trip Details")
    origin = st.text_input("Flying From (Origin Airport/City)", placeholder="Enter your departure city/airport")
    destination = st.text_input("Flying To (Destination Airport/City)", placeholder="Enter your destination city/airport")
    travel_dates = st.date_input("📅 Travel Dates", [], help="Select your trip's start and end dates.")
    budget = st.selectbox("💰 Select your budget level", ["Low (up to $5,000)", "Medium ($5,000 to $10,000)", "High ($10,000+)"])
    interests = st.multiselect("🎯 Select your interests", ["Beach", "Hiking", "Museums", "Local Food", "Shopping", "Parks", "Cultural Sites", "Nightlife"])

# Main Content Section
col1, col2 = st.columns([1, 1])

with col1:
    generate_button = st.button("📝 Generate Travel Itinerary")
with col2:
    if st.button("📊 Post-Trip Feedback"):
        st.session_state.post_trip_active = True

# Generate Itinerary Section
if generate_button:
    if not origin or not destination or len(travel_dates) != 2:
        st.error("⚠️ Please provide all required details: origin, destination, and a valid travel date range.")
    else:
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        with st.spinner("Fetching details..."):
            st.session_state.flight_prices = fetch_flight_prices(origin, destination, travel_dates[0].strftime("%Y-%m-%d"))
            st.session_state.itinerary = generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget)

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

# Post-trip section
if st.session_state.post_trip_active:
    st.header("Post-Trip Feedback & Summary")
    
    # Initialize feedback data in session state if not exists
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []
    
    # User input table for trip experience
    st.subheader("Rate Your Experience")
    parameters = [
        "Sight-seeing locations",
        "Hotels",
        "Food",
        "Local transport",
        "Local population (Friendliness, Helpfulness, Hospitable)",
        "Weather"
    ]
    
    for param in parameters:
        col1, col2 = st.columns([1, 2])
        with col1:
            rating = st.slider(f"{param} Rating (1-10)", 1, 10, 5, key=f"rating_{param}")
        with col2:
            review = st.text_input(f"Review for {param}", key=f"review_{param}")
        
        # Store in session state
        if f"feedback_{param}" not in st.session_state:
            st.session_state[f"feedback_{param}"] = {"rating": rating, "review": review}
    
    if st.button("Submit Feedback", key="submit_feedback"):
        feedback_data = [
            {
                "Parameter": param,
                "Rating": st.session_state[f"feedback_{param}"]["rating"],
                "Review": st.session_state[f"feedback_{param}"]["review"]
            }
            for param in parameters
        ]
        st.session_state.feedback_submitted = pd.DataFrame(feedback_data)
        st.success("Feedback submitted successfully!")
        st.write(st.session_state.feedback_submitted)

    # Excel input for expenses
    st.subheader("Upload Expenses (Excel File)")
    expense_file = st.file_uploader("Upload an Excel file with expenses", type=["xlsx"], key="expense_file")
    if expense_file is not None:
        try:
            expense_df = pd.read_excel(expense_file)
            st.write("Expenses from Excel:")
            st.write(expense_df)
            
            # Use LangChain for expense analysis
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
            expense_text = expense_df.to_string()
            analysis_prompt = f"Analyze these travel expenses and provide total spending, main categories, and any notable patterns: {expense_text}"
            
            if st.button("Analyze Expenses"):
                with st.spinner("Analyzing expenses..."):
                    analysis = llm.predict(analysis_prompt)
                    st.write("Expense Analysis:")
                    st.write(analysis)
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
    
    # Generate trip summary
    if st.button("Generate Trip Summary"):
        if hasattr(st.session_state, 'feedback_submitted'):
            feedback_text = "\n".join([
                f"{row['Parameter']}: Rated {row['Rating']}/10 - {row['Review']}"
                for _, row in st.session_state.feedback_submitted.iterrows()
            ])
            
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
            summary_prompt = f"""
            Based on the following feedback, create a comprehensive trip summary:
            {feedback_text}
            Please include:
            1. Overall experience highlights
            2. Areas of excellence
            3. Areas for improvement
            4. Recommendations for future travelers
            """
            
            with st.spinner("Generating summary..."):
                summary = llm.predict(summary_prompt)
                st.write("Trip Summary:")
                st.write(summary)
        else:
            st.warning("Please submit feedback first to generate a trip summary.")
    
    # Add a button to go back to main page
    if st.button("Return to Trip Planning"):
        st.session_state.post_trip_active = False
        st.experimental_rerun()
