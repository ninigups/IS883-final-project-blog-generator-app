#os.environ["OPENAI_API_KEY"] = st.secrets['TestKey1']
#os.environ["SERPER_API_KEY"] = st.secrets["SerperKey1"]

#my_secret_key = st.secrets['MyOpenAIKey']
#os.environ["OPENAI_API_KEY"] = my_secret_key

#my_secret_key = st.secrets['TestKey1']
#os.environ["OPENAI_API_KEY"] = my_secret_key

#my_secret_key = st.secrets['TestKey1']
#openai.api_key = my_secret_key

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

    # Styles for the document
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    section_style = styles["Heading2"]
    text_style = styles["BodyText"]

    elements = []

    # Add title
    elements.append(Paragraph("Travel Itinerary", title_style))
    elements.append(Spacer(1, 20))  # Add space

    # Add itinerary section
    elements.append(Paragraph("Itinerary:", section_style))
    for line in itinerary.splitlines():
        elements.append(Paragraph(line, text_style))
    elements.append(Spacer(1, 20))  # Add space

    # Add flight prices section
    elements.append(Paragraph("Flight Prices:", section_style))
    for line in flight_prices.splitlines():
        elements.append(Paragraph(line, text_style))
    elements.append(Spacer(1, 20))  # Add space

    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit UI configuration
st.set_page_config(
    page_title="Travel Planning Assistant",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for sky blue background
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

# Store results in session state
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "flight_prices" not in st.session_state:
    st.session_state.flight_prices = None

# Main Content Section
# Initialize session state for post-trip
if 'post_trip_active' not in st.session_state:
    st.session_state.post_trip_active = False

# Main buttons layout
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📝 Generate Travel Itinerary", use_container_width=True):
        if not origin or not destination or len(travel_dates) != 2:
            st.error("⚠️ Please provide all required details: origin, destination, and a valid travel date range.")
        else:
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate loading time
                progress.progress(i + 1)

            with st.spinner("Fetching details..."):
                st.session_state.flight_prices = fetch_flight_prices(origin, destination, travel_dates[0].strftime("%Y-%m-%d"))
                st.session_state.itinerary = generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget)

with col2:
    if st.button("📊 Post-Trip Feedback", use_container_width=True):
        st.session_state.post_trip_active = True
# Display results only if available
if st.session_state.itinerary and st.session_state.flight_prices:
    st.success("✅ Your travel details are ready!")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(display_card("Itinerary", st.session_state.itinerary), unsafe_allow_html=True)

    with col2:
        st.markdown(display_card("Flight Prices", st.session_state.flight_prices), unsafe_allow_html=True)

    # Display map links directly on the main page
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

    # Generate and provide download link for PDF
    pdf_buffer = create_pdf(st.session_state.itinerary, st.session_state.flight_prices)
    st.download_button(
        label="📥 Download Itinerary as PDF",
        data=pdf_buffer,
        file_name="travel_itinerary.pdf",
        mime="application/pdf",
    )
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Initialize post-trip state if not exists
if "post_trip_active" not in st.session_state:
    st.session_state.post_trip_active = False

if st.session_state.post_trip_active:
    st.header("Post-Trip Feedback & Summary")
    
    # Location, date and duration visited
    col1, col2, col3 = st.columns(3)
    with col1:
        location_visited = st.text_input("Location Visited", placeholder="Enter city/country")
    with col2:
        date_visited = st.date_input("Date Visited")
    with col3:
        duration = st.number_input("Duration (days)", min_value=1, value=1, step=1)
    
    # Initialize LLM 
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
    
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
    
    # Initialize feedback data storage
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []

    # Collect ratings and reviews
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
        if not location_visited:
            st.error("Please enter the location visited.")
        else:
            # Create DataFrame with collected feedback
            feedback_df = pd.DataFrame(feedback_data)
            feedback_df["Location"] = location_visited
            feedback_df["Date"] = date_visited
            feedback_df["Duration"] = f"{duration} days"
            
            # Generate blog-style summary
            all_reviews = "\n".join([f"{row['Parameter']}: {row['Rating']}/10 - {row['Review']}" 
                                   for row in feedback_data])
            
            blog_prompt = f"""
            Write a personal blog-style travel review based on your {duration}-day visit to {location_visited} on {date_visited}:
            
            {all_reviews}
            
            Write in first person, make it engaging and personal, highlight both positives and negatives,
            and make it feel like a genuine travel blog post. Keep it to 2-3 paragraphs.
            Include an overall sentiment analysis (Positive/Negative/Neutral) at the end.
            """
            
            with st.spinner("Generating your travel blog..."):
                blog_post = llm.predict(blog_prompt)
                st.subheader("Your Travel Story")
                st.write(blog_post)

            # Hidden Gems section
            st.subheader("💎 Planning to Visit Again? Discover Hidden Gems")
            with st.spinner("Finding unique local spots..."):
                hidden_gems_query = f"hidden gems OR secret spots OR local favorites OR off the beaten path {location_visited} -tripadvisor -tourradar"
                try:
                    search_results = serper_tool.func(hidden_gems_query)
                    
                    hidden_gems_prompt = f"""
                    Based on these search results about {location_visited}, provide:
                    
                    1. Lesser-Known Local Spots: Hidden restaurants, cafes, or viewpoints that tourists often miss
                    2. Authentic Local Experiences: Unique cultural activities or traditions you can participate in
                    3. Local Tips: Best times to visit these places and insider recommendations
                    
                    Focus on unique, authentic experiences that aren't in typical tourist guides.
                    Make it personal by addressing the reader directly using "you" and "your".
                    Format with clear, concise headings and bullet points.
                    Keep the text size consistent throughout.
                    Do not add location name in the title, start directly with the categories.
                    """
                    
                    hidden_gems_info = llm.predict(hidden_gems_prompt)
                    st.write(hidden_gems_info)
                except Exception as e:
                    st.error(f"Error finding hidden gems: {str(e)}")

            # Recommendations section 
            st.subheader("🌍 Recommended Destinations for Your Next Visit")
            with st.spinner("Finding personalized recommendations..."):
                recommendation_prompt = f"""
                Based on your ratings and feedback for {location_visited}:
                {all_reviews}
                
                Suggest 3 destinations that align with your ratings and preferences. For each destination, write:
                1. Why it matches your preferences: (emphasize by using "you" and "your")
                2. Best time to visit
                3. Estimated budget needed: Always format as $XX-$XX USD per day (use numbers and hyphen)
                
                Format as clear sections for each destination.
                Use "you" and "your" throughout the recommendations.
                For all budget ranges, use format like $80-$120 per day.
                Do not add any concluding statements.
                """
                
                try:
                    recommendations = llm.predict(recommendation_prompt)
                    st.write(recommendations)
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
