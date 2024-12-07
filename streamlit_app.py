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
    generate_button = st.button("📝 Generate Travel Itinerary", use_container_width=True)
with col2:
    post_trip_button = st.button("📊 Post-Trip Feedback", use_container_width=True)

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
    
    # Add location and date visited fields
    col1, col2 = st.columns(2)
    with col1:
        location_visited = st.text_input("Location Visited", placeholder="Enter city/country")
    with col2:
        date_visited = st.date_input("Date Visited")
    
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
    
    feedback_data = []
    for param in parameters:
        st.write(f"**{param}**")
        col1, col2 = st.columns([1, 2])
        with col1:
            rating = st.slider(f"Rating (1-10)", 1, 10, 5, key=f"rating_{param}")
        with col2:
            review = st.text_area(f"Your thoughts about {param}", key=f"review_{param}", height=100)
        feedback_data.append({
            "Parameter": param,
            "Rating": rating,
            "Review": review
        })
    
    if st.button("Submit Feedback", key="submit_feedback"):
        # Include location and date in feedback
        feedback_df = pd.DataFrame(feedback_data)
        feedback_df["Location"] = location_visited
        feedback_df["Date"] = date_visited
        
        st.session_state.feedback_submitted = feedback_df
        st.success("Feedback submitted successfully!")

        # Generate blog-style summary
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
        
        blog_prompt = f"""
        Write a personal blog-style travel review based on this feedback for {location_visited} visited on {date_visited}:
        
        {', '.join([f"{row['Parameter']}: {row['Rating']}/10 - {row['Review']}" for row in feedback_data])}
        
        Write in first person, make it engaging and personal, highlight both positives and negatives,
        and make it feel like a genuine travel blog post. Keep it to 2-3 paragraphs.
        """
        
        with st.spinner("Generating your travel blog..."):
            blog_post = llm.predict(blog_prompt)
            st.subheader("Your Travel Story")
            st.write(blog_post)

        # Travel companion search section
        st.subheader("🤝 Looking for Travel Companions?")
        if st.button("Find Travel Companions"):
            companion_query = f"site:reddit.com travel companion {location_visited} OR travel buddy {location_visited}"
            try:
                search_results = serper_tool.func(companion_query)
                
                companion_prompt = f"""
                Based on these search results about travel companions:
                {search_results}
                
                Please summarize:
                1. Popular platforms/communities for finding travel companions
                2. Common safety tips for traveling with new people
                3. Recommended ways to connect with potential travel buddies
                4. Current travel companion opportunities for {location_visited}
                
                Format it in a clear, easy-to-read way.
                """
                
                with st.spinner("Finding travel companion information..."):
                    companion_info = llm.predict(companion_prompt)
                    st.write(companion_info)
            except Exception as e:
                st.error(f"Error searching for travel companions: {str(e)}")

        # Expense Analysis Section
        st.subheader("Upload Expenses (Excel File)")
        expense_file = st.file_uploader("Upload an Excel file with expenses", type=["xlsx"], key="expense_file")
        if expense_file is not None:
            try:
                expense_df = pd.read_excel(expense_file)
                st.write("Expenses from Excel:")
                st.write(expense_df)
                
                expense_text = expense_df.to_string()
                analysis_prompt = f"""
                Analyze these travel expenses for {location_visited} and provide:
                1. Total spending
                2. Breakdown by category
                3. Cost-saving opportunities
                4. Comparison to typical expenses for this destination
                """
                
                if st.button("Analyze Expenses"):
                    with st.spinner("Analyzing expenses..."):
                        analysis = llm.predict(analysis_prompt)
                        st.write("Expense Analysis:")
                        st.write(analysis)
                        
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
