# %%
import streamlit as st
import requests
from geopy.geocoders import Nominatim
import numpy as np
import pickle
from datetime import datetime, timedelta

# === Configuration ===
OPENWEATHER_API_KEY = "a0d558e1b239783fa28304f20f3dc7af"
# Define a fixed historical period for NASA POWER data if needed
# For this example, we'll use the last year from today for historical data
# You can adjust this based on what your model expects
end_date_historical = datetime.now().strftime("%Y%m%d")
start_date_historical = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")


# === Get Coordinates from Address ===
def get_coordinates(address):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# === Get Current Weather Data from OpenWeatherMap ===
def get_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url)
        res.raise_for_status() # Raise an exception for bad status codes
        data = res.json()
        # Safely access nested keys
        temp = data.get("main", {}).get("temp")
        humidity = data.get("main", {}).get("humidity")
        # OpenWeatherMap's rain data can be nested under 'rain' and then a time period like '1h' or '3h'
        # Use .get with a default of {} to handle cases where 'rain' key might be missing
        rainfall = data.get("rain", {}).get("1h", 0) # Default to 0 if no rain data is available
        return {
            "Temperature (°C)": temp,
            "Humidity (%)": humidity,
            "Rainfall (mm)": rainfall
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        st.error(f"Error parsing weather data: Missing key {e}")
        return None

# === Get Historical Data from NASA POWER API ===
# This function is included if you need historical data for your model,
# but based on the requested input format [temp, humidity, ph, rainfall],
# the model likely uses current or aggregated current conditions.
# If your model truly needs historical data, you'll need to process the
# time series data returned by this API (e.g., calculate averages, min/max).
# For now, we'll fetch it but primarily use current data for the feature array.
def get_historical_data(lat, lon, start_date, end_date):
    # Note: POWER API parameters are case-sensitive. 'PRECTOT' is correct.
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOT&community=ag&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        # Safely access parameters, using .get() with a default empty dictionary
        parameters = data.get('properties', {}).get('parameter', {})
        temperature_historical = parameters.get('T2M')
        rainfall_historical = parameters.get('PRECTOT')

        # You might need to process this historical data (e.g., calculate averages)
        # depending on what your model expects as input.
        # For this example, we'll just return the raw dictionaries.
        return temperature_historical, rainfall_historical

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical weather data: {e}")
        return None, None
    except KeyError as e:
        st.error(f"Error parsing historical weather data: Missing key {e}")
        return None, None


# === Load the Model ===
# Ensure 'RandomForest.pkl' is in the same directory or provide the full path
try:
    with open('RandomForest.pkl', 'rb') as f:
        model = pickle.load(f)
    st.sidebar.success("Model loaded successfully.")
except FileNotFoundError:
    st.sidebar.error("Error: 'RandomForest.pkl' model file not found.")
    model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None


# === Streamlit UI ===
st.title("🌾 Real-time Crop Predictor")
st.write("Enter a location to get weather and soil data for crop prediction.")

address = st.text_input("Enter location (e.g., Ibadan, Nigeria):")

# Process only if an address is entered
if address:
    # Add a spinner while processing
    with st.spinner(f"Fetching data for {address}..."):
        lat, lon = get_coordinates(address)

        if lat is not None and lon is not None:
            st.success(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")

            # Get current weather data
            weather_data = get_weather_data(lat, lon)


            # Get historical data (optional, depending on model requirements)
            # temperature_historical, rainfall_historical = get_historical_data(lat, lon, start_date_historical, end_date_historical)
            # if temperature_historical or rainfall_historical:
            #    st.subheader("📜 Historical Weather Data (Last Year)")
                # Display or process historical data here if needed
            #    st.write("Historical data fetched (processing depends on model needs).")


            # Display fetched data
            if weather_data is not None: # Ensure both weather and pH data were successfully retrieved
                st.subheader("📊 Fetched Data")
                st.write("Current Weather:")
                for key, value in weather_data.items():
                    st.write(f"- {key}: {value}")

                # === Create Feature Array for Prediction ===
                # Ensure all required features are available and are not None
                temp_now = weather_data.get("Temperature (°C)")
                humidity_now = weather_data.get("Humidity (%)")
                rainfall_now = weather_data.get("Rainfall (mm)") # Use current rainfall for the feature

                if temp_now is not None and humidity_now is not None and rainfall_now is not None:
                    # Arrange features in the required order: temp, humidity, rainfall
                    features = np.array([[temp_now, humidity_now, rainfall_now]])

                    st.subheader("🤖 Prediction")
                    # === Make Prediction ===
                    if model:
                        try:
                            prediction = model.predict(features)
                            st.success(f"Predicted Crop: **{prediction[0]}**") # Assuming prediction returns an array
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                    else:
                        st.warning("Model not loaded. Cannot make prediction.")
                else:
                    st.warning("Missing data required for prediction. Please check location.")

            elif weather_data is not None:
                 st.warning("Incomplete data fetched. Cannot make prediction.")


        else:
            st.error("Couldn't find the location coordinates.")

# Add instructions on how to run
st.sidebar.subheader("How to Run")
st.sidebar.write("1. Save the code as a Python file (e.g., `app.py`).")
st.sidebar.write("2. Ensure you have `streamlit`, `requests`, `geopy`, `numpy`, and `scikit-learn` (or whatever library your model uses) installed (`pip install streamlit requests geopy numpy scikit-learn`).")
st.sidebar.write("3. Place your trained model file (`RandomForest.pkl`) in the same directory as the Python file.")
st.sidebar.write("4. Run the app from your terminal using `streamlit run app.py`.")
st.sidebar.write("5. Enter a location in the text box.")
