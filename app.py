import pickle
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
LOW_SALES, HIGH_SALES = 190, 285    # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption(
    "Select a city and date to predict ice cream sales using "
    "temperature, rain probability, and whether it is a weekend."
)

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=15),   # Allow dates up to 15 days ahead
)

if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast
    latitude, longitude = CITIES[city]

    try:
        response = requests.get(
            URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": str(selected_date),
                "end_date": str(selected_date),
            },
            timeout=30,
        )
        response.raise_for_status()

    except requests.RequestException:
        st.error(
            "The weather service is not responding. "
            "Please wait a minute and try again."
        )
        st.stop()

    temperature = response.json()["daily"]["temperature_2m_max"][0]


    rain_chance = response.json()["daily"]["precipitation_probability_max"][0]


    if temperature is None or rain_chance is None:
        st.error(
            "The forecast for this date is incomplete. "
            "Please select another date or city."
        )
        st.stop()


    is_weekend = int(selected_date.weekday() >= 5)

    # 2. Give the model all three predictors
    features = pd.DataFrame({
        "temperature": [temperature],
        "rain_chance": [rain_chance],
        "is_weekend": [is_weekend]
    })

    prediction = max(0, model.predict(features)[0])

    # 3. Show the result
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{temperature:.0f} °F")
    col2.metric("Rain chance", f"{rain_chance:.0f}%")
    col3.metric("Day type", "Weekend" if is_weekend else "Weekday")
    col4.metric("Predicted sales", f"{prediction:.0f} cones")


    st.caption(
        "Predictions are based on simulated training data. "
        "Slow day: below 190 cones. "
        "Normal day: 190–285 cones. "
        "Busy day: above 285 cones. "
        "These estimates are for educational purposes."
    )


    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(
            f"This temperature is outside the range the model learned from "
            f"({LOW_TEMP} to {HIGH_TEMP} °F). Treat the number with caution."
        )


    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction >= LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")


