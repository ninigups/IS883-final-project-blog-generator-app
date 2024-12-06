#os.environ["OPENAI_API_KEY"] = st.secrets['TestKey1']
#os.environ["SERPER_API_KEY"] = st.secrets["SerperKey1"]

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
import easyocr

# Load API keys from st.secrets directly
openai.api_key = st.secrets["TestKey1"]
serper_api_key = st.secrets["SerperKey1"]
os.environ["SERPER_API_KEY"] = serper_api_key

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
        categories, or a table wherever appropriate. Include key highlights like the cheapest fare, airlines, and travel dates.
        Ignore missing or irrelevant text.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # updated model
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
            model="gpt-3.5-turbo",  # updated model
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

# Streamlit UI configuration
st.set_page_config(
    page_title="Travel Planning Assistant",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sky blue background
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

# Main App Title
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

# Session state for itinerary and flight prices
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "flight_prices" not in st.session_state:
    st.session_state.flight_prices = None

# Generate Travel Itinerary Button
if st.button("📝 Generate Travel Itinerary"):
    if not origin or not destination or len(travel_dates) != 2:
        st.error("⚠️ Please provide origin, destination, and a valid travel date range.")
    else:
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        with st.spinner("Fetching details..."):
            st.session_state.flight_prices = fetch_flight_prices(origin, destination, travel_dates[0].strftime("%Y-%m-%d"))
            st.session_state.itinerary = generate_itinerary_with_chatgpt(origin, destination, travel_dates, interests, budget)

# Display itinerary and flight prices if available
if st.session_state.itinerary and st.session_state.flight_prices:
    st.success("✅ Your travel details are ready!")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(display_card("Itinerary", st.session_state.itinerary), unsafe_allow_html=True)

    with col2:
        st.markdown(display_card("Flight Prices", st.session_state.flight_prices), unsafe_allow_html=True)

    # Show map links
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

# Post-Travel Section 
show_post_travel = st.checkbox("Show Post-Travel Section")
if show_post_travel:
    st.header("Post-Travel Review")
    st.subheader("Upload Receipts for Expense Extraction")

    receipt_files = st.file_uploader("Upload receipts (PDF or Images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

import easyocr
reader = easyocr.Reader(['en'])

preprocess_contrast = st.slider("Increase Image Contrast Factor", 1.0, 3.0, 1.0, 0.1)
st.write("Use a higher factor if the receipt text is faint.")

def ocr_image(img, contrast_factor=1.0):
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    import numpy as np
    # Use easyocr directly
    result = reader.readtext(np.array(img), detail=0)
    text = "\n".join(result)
    return text


    preprocess_contrast = st.slider("Increase Image Contrast Factor", 1.0, 3.0, 1.0, 0.1)
    st.write("Increase contrast if the receipt text is faint.")

    improved_patterns = [
        r"(?i)(total\s*:?\s*\$?\s*([\d.,]+))",
        r"(?i)(grand\s*total\s*:?\s*\$?\s*([\d.,]+))",
        r"(?i)(amount\s*due\s*:?\s*\$?\s*([\d.,]+))",
        r"(?i)(tip\s*:?\s*\$?\s*([\d.,]+))"
    ]

    extracted_expenses = []

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

    from PIL import Image, ImageEnhance
    import pytesseract
    import PyPDF2

    def ocr_image(img, contrast_factor=1.0):
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        if ocr_engine_choice == "pytesseract":
            text = pytesseract.image_to_string(img)
        else:
            # If easyocr is available and successfully imported
            if 'easyocr' in globals():
                import numpy as np
                result = reader.readtext(np.array(img), detail=0)
                text = "\n".join(result)
            else:
                text = "easyocr selected but not available."
        return text

    text_content = ""  
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
                st.write("No amounts found. Try adjusting contrast or using a different OCR engine.")

    if extracted_expenses:
        st.subheader("Extracted Expenses")
        st.write(extracted_expenses)

    st.subheader("Classify Expenses")

    from langchain.agents import initialize_agent, Tool
    from langchain.agents.agent_types import AgentType
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    tools = [serper_tool]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    if extracted_expenses:
        first_expense_text = text_content  # Use last processed text
        user_query = f"Classify this expense into a category like Food, Travel, Accommodation, Entertainment, Miscellaneous, or Grand total. The expense text is: {first_expense_text}"

        if st.button("Classify Expense Using Agent"):
            with st.spinner("The agent is reasoning..."):
                agent_answer = agent.run(user_query)
            st.write("**Agent's Answer:**")
            st.write(agent_answer)
    else:
        st.write("No expenses to classify yet. Upload receipts above.")
