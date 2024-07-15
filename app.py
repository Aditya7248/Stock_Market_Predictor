import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the pre-trained neural network model
@st.cache_data
def load_nn_model():
    return load_model(r'C:\Desktop\Stock_Market_Prediction\Stock_Preduction_Model.h5')

def main():
    st.title('Stock Market Predictor')

    # User input for stock symbol
    stock = st.text_input('Enter Stock Symbol', 'NVDA')

    # Date range for fetching stock data
    start_date = st.date_input('Start Date', pd.to_datetime('2013-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-03-30'))

    # Fetching stock data
    data = yf.download(stock, start_date, end_date)
    st.subheader('Stock Data')
    st.write(data)

    if data.empty:
        st.error("No data found for the specified stock symbol and date range. Please adjust your inputs.")
        return

    # Preprocessing data
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    if data_train.empty or data_test.empty:
        st.error("Error in splitting the data into training and testing sets. Please check your data.")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

    if data_test.empty:
        st.error("Error in concatenating the data. Please check your data.")
        return

    data_test_scale = scaler.fit_transform(data_test)

    if len(data_test_scale) == 0:
        st.error("Error: No data available for scaling.")
        return

    # Visualizations
    st.subheader('Price vs Moving Averages')
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(ma_50_days, 'r', label='MA50')
    axs[0].plot(data.Close, 'g', label='Price')
    axs[0].legend()

    axs[1].plot(ma_100_days, 'b', label='MA100')
    axs[1].plot(ma_50_days, 'r', label='MA50')
    axs[1].plot(data.Close, 'g', label='Price')
    axs[1].legend()

    axs[2].plot(ma_200_days, 'b', label='MA200')
    axs[2].plot(ma_100_days, 'r', label='MA100')
    axs[2].plot(data.Close, 'g', label='Price')
    axs[2].legend()

    st.pyplot(fig)

    # Load the neural network model
    model = load_nn_model()

    # Prepare data for prediction
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    if len(x) == 0 or len(y) == 0:
        st.error("Error: Insufficient data for prediction.")
        return

    # Make predictions
    predictions = model.predict(x)

    # Scale back the predictions
    predictions = predictions * scaler.scale_[0]
    y = y * scaler.scale_[0]

    # Plot original and predicted prices
    st.subheader('Original Price vs Predicted Price')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predictions, 'r', label='Predicted Price')
    ax.plot(y, 'g', label='Original Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()

## streamlit run c:/Users/tiwar/OneDrive/Desktop/Stock_Market_Prediction/app.py (run website terminal)
