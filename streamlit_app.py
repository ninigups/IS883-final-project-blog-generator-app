#my_secret_key = st.secrets['MyOpenAIKey']
#os.environ["OPENAI_API_KEY"] = my_secret_key

#my_secret_key = st.secrets['TestKey1']
#os.environ["OPENAI_API_KEY"] = my_secret_key

#my_secret_key = st.secrets['TestKey1']
#openai.api_key = my_secret_key

import streamlit as st
import openai
import pandas as pd
import urllib.parse
import re
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Load your API Key
my_secret_key = st.secrets['TestKey1']
openai.api_key = my_secret_key

# Function to extract activities and coordinates from the itinerary
def extract_activities_with_coordinates(itinerary_text):
    # Match activities with city and latitude/longitude
    activity_pattern = re.compile(
        r"Activity Name: (.*?)\nCity and Country: (.*?)\n.*?Latitude & Longitude: ([\d.\-]+), ([\d.\-]+)",
        re.DOTALL
    )
    activities = []
    for match in activity_pattern.finditer(itinerary_text):
        place, city, lat, lon = match.groups()
        activities.append({
            'Place': place.strip(),
            'City': city.strip(),
            'lat': float(lat.strip()),
            'lon': float(lon.strip())
        })
    return pd.DataFrame(activities)

# Fallback function to geocode place names
def geocode_places(places, context="Delhi, India"):
    geolocator = Nominatim(user_agent="travel_planner")
    geocoded_data = []
    for place in places:
        try:
            # Add city and country context to the place name
            full_address = f"{place}, {context}"
            location = geolocator.geocode(full_address, timeout=10)
            if location:
                geocoded_data.append({'Place': place, 'lat': location.latitude, 'lon': location.longitude})
            else:
                st.warning(f"Could not geocode: {place}")
        except GeocoderTimedOut:
            st.warning(f"Geocoding timed out for {place}. Skipping.")
    return pd.DataFrame(geocoded_data)

# Initialize session state for navigation if not already set
if "active_branch" not in st.session_state:
    st.session_state.active_branch = None  # None means no branch is active

st.header("Travel Planning Assistant 🛫")
st.subheader("Choose an option to get started:")

# Display buttons only if no branch is active
if st.session_state.active_branch is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Pre-travel", key="pre_travel_btn"):
            st.session_state.active_branch = "Pre-travel"  # Set active branch to Pre-travel

    with col2:
        if st.button("Post-travel", key="post_travel_btn"):
            st.session_state.active_branch = "Post-travel"  # Set active branch to Post-travel

# Pre-travel Branch
if st.session_state.active_branch == "Pre-travel":
    st.header("Plan Your Travel 🗺️")
    origin = st.text_input("Flying From (Origin Airport/City)")
    destination = st.text_input("Flying To (Destination Airport/City)")
    travel_dates = st.date_input("Select your travel dates", [])
    
    budget = st.selectbox("Select your budget level", ["Low (up to $5,000)", "Medium ($5,000 to $10,000)", "High ($10,000+)"])
    generate_itinerary = st.button("Generate Itinerary")

    if generate_itinerary:
        # Improved prompt with clear request for location and context information
        prompt_template = """
        You are a travel assistant. Create a detailed itinerary for a trip from {origin} to {destination}. 
        The user is interested in general activities. The budget level is {budget}. 
        The travel dates are {travel_dates}. For each activity, include:
        - Activity name
        - City and country context
        - Latitude and longitude for geocoding purposes
        Provide a minimum of 5 activities with full details for accurate location mapping.
        """
        prompt = prompt_template.format(origin=origin, destination=destination, budget=budget, travel_dates=travel_dates)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            itinerary = response.choices[0].message["content"]
            
            st.subheader("Generated Itinerary:")
            st.write(itinerary)

            # Extract activities with coordinates
            activity_df = extract_activities_with_coordinates(itinerary)

            if not activity_df.empty:
                st.subheader("Map of Activities:")
                st.map(activity_df[['lat', 'lon']])
            else:
                st.write("No activities with coordinates found. Attempting to geocode...")
                activities = re.findall(r"Activity Name: (.*?)\nCity and Country: (.*?)\n", itinerary)
                geocoded_df = geocode_places(
                    [f"{place}, {city}" for place, city in activities]
                )
                if not geocoded_df.empty:
                    st.map(geocoded_df[['lat', 'lon']])
                else:
                    st.write("Could not geocode any activities.")

        except Exception as e:
            st.error(f"An error occurred while generating the itinerary: {e}")

# Post-travel Branch
elif st.session_state.active_branch == "Post-travel":
    st.header("Post-travel: Data Classification and Summary")
    uploaded_file = st.file_uploader("Upload your travel data (Excel file)", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("Data Preview:")
        st.write(df.head())

# Add a Back Button
if st.session_state.active_branch is not None:
    if st.button("Back to Home", key="back_btn"):
        st.session_state.active_branch = None  # Reset active branch
import pytesseract
from PIL import Image
import PyPDF2
import io

if 'active_branch' in st.session_state:
    # Post-travel Branch Additions
    if st.session_state.active_branch == "Post-travel":
        # Expense Data from Excel
        st.header("Post-travel: Data Classification and Summary")
        st.subheader("Upload Expenses (Excel File)")
        expense_file = st.file_uploader("Upload your expenses (Excel file)", type=["xlsx"], key="expense_file")
        if expense_file is not None:
            expense_df = pd.read_excel(expense_file)
            st.subheader("Expenses Data from Excel:")
            st.write(expense_df)
        else:
            expense_df = pd.DataFrame()  

        # Expense Data from Receipts (Images and PDFs)
        st.subheader("Upload Expense Receipts (Images or PDFs)")
        receipt_files = st.file_uploader(
            "Upload your expense receipts (images or PDFs)",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key="receipt_files"
        )
        expenses = []
        if receipt_files:
            try:
                for receipt_file in receipt_files:
                    file_type = receipt_file.type
                    if file_type == "application/pdf":
                        # Process PDF file
                        pdf_reader = PyPDF2.PdfReader(receipt_file)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()
                    
                        amount_matches = re.findall(r"(?i)(total|amount due|grand total)[^\d]*(\d+\.\d{2})", text)
                        if amount_matches:
                            for match in amount_matches:
                                amount = float(match[1])
                                expenses.append({
                                    "File": receipt_file.name,
                                    "Amount": amount,
                                    "Text": text
                                })
                    else:
                        # Image File Processing
                        image = Image.open(receipt_file)
                        text = pytesseract.image_to_string(image)
                        amount_matches = re.findall(r"(?i)(total|amount due|grand total)[^\d]*(\d+\.\d{2})", text)
                        if amount_matches:
                            for match in amount_matches:
                                amount = float(match[1])
                                expenses.append({
                                    "File": receipt_file.name,
                                    "Amount": amount,
                                    "Text": text
                                })
                if expenses:
                    expenses_df = pd.DataFrame(expenses)
                    st.subheader("Extracted Expenses from Receipts:")
                    st.write(expenses_df)
                else:
                    st.write("No expenses extracted from receipts.")
                    expenses_df = pd.DataFrame()  # Empty DataFrame if no expenses extracted
            except Exception as e:
                st.error(f"An error occurred while processing receipts: {e}")
                expenses_df = pd.DataFrame()
        else:
            expenses_df = pd.DataFrame()

        # Consolidate Expenses
        if not expense_df.empty and not expenses_df.empty:
            combined_expenses_df = pd.concat([expense_df, expenses_df], ignore_index=True)
            st.subheader("Consolidated Expenses:")
            st.write(combined_expenses_df)
        elif not expense_df.empty:
            st.subheader("Expenses (from Excel):")
            st.write(expense_df)
            combined_expenses_df = expense_df
        elif not expenses_df.empty:
            st.subheader("Expenses (from Receipts):")
            st.write(expenses_df)
            combined_expenses_df = expenses_df
        else:
            st.write("No expenses data available.")
            combined_expenses_df = pd.DataFrame()

        # Trip Experience Feedback
        st.header("Trip Experience Feedback")
        st.write("Please provide your ratings and reviews for the following parameters:")

        # Define parameters
        parameters = [
            "Sight-seeing locations",
            "Hotels",
            "Food",
            "Local transport",
            "Local population (friendliness, helpfulness, hospitable)",
            "Weather"
        ]
        feedback_data = []

        for param in parameters:
            st.subheader(param)
            rating = st.slider(f"Rating for {param} (1-10)", min_value=1, max_value=10, key=f"rating_{param}")
            review_text = st.text_area(f"Review for {param}", key=f"review_{param}")
            feedback_data.append({
                "Parameter": param,
                "Rating (1-10)": rating,
                "Review Text": review_text
            })

        if st.button("Submit Feedback"):
            feedback_df = pd.DataFrame(feedback_data)
            st.subheader("Your Trip Experience Feedback:")
            st.write(feedback_df)
