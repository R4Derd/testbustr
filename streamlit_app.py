import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load data
@st.cache_data
def load_data():
    url = "https://data.bus-data.dft.gov.uk/api/v1/gtfsrtdatafeed/?boundingBox"  # Replace with the actual URL
    data = pd.read_csv(url)
    return data

data = load_data()

# Step 2: Preprocess data
def preprocess_data(data):
    # Handle missing values, encode categorical variables, etc.
    data = data.dropna()
    # Example feature engineering
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
    return data

data = preprocess_data(data)

# Step 3: Feature selection
features = ['hour', 'day_of_week', 'route_id', 'bus_id', 'stop_id']
X = data[features]
y = data['delay']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Build Streamlit app
st.title("Bus Delay Prediction App")

# Sidebar for user input
st.sidebar.header("Input Features")
hour = st.sidebar.slider("Hour of the day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of the Week", [0, 1, 2, 3, 4, 5, 6])
route_id = st.sidebar.text_input("Route ID", "1")
bus_id = st.sidebar.text_input("Bus ID", "1001")
stop_id = st.sidebar.text_input("Stop ID", "500")

user_input = pd.DataFrame({
    'hour': [hour],
    'day_of_week': [day_of_week],
    'route_id': [route_id],
    'bus_id': [bus_id],
    'stop_id': [stop_id]
})

# Predict delay
prediction = model.predict(user_input)

st.write("## Predicted Delay")
st.write(f"{prediction[0]:.2f} minutes")

# Show data if checkbox is selected
if st.checkbox("Show raw data"):
    st.write(data)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("## Model Evaluation")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")
