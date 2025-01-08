import streamlit as st
import pandas as pd
from datetime import date, timedelta
# import pymysql
from sqlalchemy import create_engine
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import os
import matplotlib.pyplot as plt


def train_real(df, df_normalized, stock_label):
    if "normalized_data" not in st.session_state and "filtered_data"  not in st.session_state:
        st.error("Data belum tersedia.")
        return

    with st.spinner('Initial Dataset Training'):
        X = []
        y = []
        #Windowing
        window_size = 30
        for i in range(window_size, len(df_normalized)):
            X.append(df_normalized[i-window_size:i])
            y.append(df_normalized[i, :])
        X, y = np.array(X), np.array(y)

        #Splitting
        split = int(len(X) * 0.8)

        # Split the dataset
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create an index for plotting
        dates = df.index[window_size:]  # Adjust dates to match the X dataset

        # Build the model
        model = Sequential()

        # Add the Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dense(5))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_absolute_error')

        
        # Create a separate scaler for all attributes (multi-output)
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)


        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Predict on test set
        predicted = model.predict(X_test)

        # Inverse-transform predictions and y_test for all features
        predicted_stock_prices = feature_scaler.inverse_transform(predicted)
        y_test_actual = feature_scaler.inverse_transform(y_test)

        # Plot the 'Close' price (as an example) from predictions and actual values
        test_dates = df.index[len(X_train) + window_size:len(X_train) + window_size + len(y_test)]

        plt.figure(figsize=(10, 6))
        plt.plot(test_dates, y_test_actual[:, df.columns.get_loc('Close') - 1], color='blue', label='Actual Close Price')
        plt.plot(test_dates, predicted_stock_prices[:, df.columns.get_loc('Close') - 1], color='red', label='Predicted Close Price')
        plt.title('Stock Price Prediction (Close)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        folder_path = os.path.join(os.getcwd(), "model/predict_model")
        model.save(f'{folder_path}/{stock_label}-model.h5')
  
    st.success("Training Complete!")


def predict_real():
    if "selected_table" in st.session_state:
        selected_table = st.session_state["selected_table"]
    else:
        st.error("Data saham belum tersedia. Silakan kembali ke halaman sebelumnya untuk memilih data.")
        st.stop()

    selected_table = st.session_state["selected_table"]  

    st.markdown(
        """
        <h1 style="text-align: center;">Prediksi Harga Saham</h1>
        """,
        unsafe_allow_html=True
    )
    
    db = "mysql+pymysql://root:lionel123@localhost:3306/skripsi"
    # Koneksi ke database
    db_connection = create_engine(db)   
    conn = db_connection.connect()

    # Membaca data terbaru dari database
    query = f"""
        SELECT * 
        FROM {selected_table} 
        ORDER BY Date ASC
    """

    # Untuk Inisial Train
    df_all = pd.read_sql(query, conn, index_col=['Date'])
    df_all.columns = df_all.columns.str.strip()
    predict_date = df_all.index[-1] + pd.Timedelta(days=1)

    st.divider()
    st.text(f"Prediksi harga Saham {st.session_state['selected_stock']}, prediksi akan dimulai dari tanggal {predict_date.date()}")
    st.text(f"Tentukan Jumlah Hari Prediksi:")

    # Input jumlah hari prediksi
    prediction_days = st.slider("Jumlah Hari Prediksi", min_value=1, max_value=30, value=7)
    



    # Plotting
    st.line_chart(df_all['Close'])
    # plt.figure(figsize=(10, 5))
    # plt.plot(df_all.index, df_all['Close'], label='Dataset', color='blue')
    # plt.title('Dataset Predict Start Date')
    # plt.xlabel('Date')
    # plt.ylabel('Close Price')
    # plt.legend()
    # st.pyplot(plt)

    # Membaca data dari database
    query = f"""
        SELECT * 
        FROM {selected_table}
        ORDER BY Date DESC
        LIMIT 30
    """
    df = pd.read_sql(query, conn, index_col=['Date'])
    df.columns = df.columns.str.strip()
    df.sort_index(inplace=True)

    
    # Tombol untuk memulai prediksi
    if st.button("Prediksi Harga",use_container_width=True):
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        df_all_scaled = scaler.fit_transform(df_all)

        predicted_prices = [[]]     
        last_input = df_scaled     
        # st.write(last_input)
        
        folder_path = os.path.join(os.getcwd(), "model/predict_model")
        model_path = f"{folder_path}/{selected_table}-model.h5"

        #Inisial Training
        if not os.path.exists(model_path):
            train_real(df_all, df_all_scaled, selected_table)

        # Memuat model
        model = load_model(model_path)  # Ganti dengan path model Anda


        # print(last_input.shape)
        for _ in range(prediction_days):  
            pred = model.predict(last_input.reshape(1, 30, 5))
            predicted_row = pred.flatten()  # Extract the predicted row (all features)
            
            # Save the predicted Close price (4th column) for later use
            predicted_prices[0].append(predicted_row[3])  # Append the Close price only
            
            # Append the entire predicted row to the input
            last_input = np.vstack([last_input[1:], predicted_row]) # Remove the first row and append the predicted row


        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices = df[['Close']].values
        close_scaler.fit(close_prices)
        predicted_prices_actual = close_scaler.inverse_transform(predicted_prices)
        flatten_predicted = predicted_prices_actual.flatten()
        flatten_predicted = flatten_predicted.astype(int)

        # Get the last date from the dataset
        last_date = df_all.index[-1]  # Assuming the index is of type DateTimeIndex

        # Generate the next `prediction_days` dates starting from the last date
        predicted_dates = pd.date_range(last_date, periods=prediction_days + 1, freq='D')[1:]  # Skip the last date, as it's already in the dataset

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(predicted_dates, flatten_predicted, marker='o', label='Predicted Close Prices', color='blue')

        # Set the title and labels
        plt.title(f"Predicted Close Prices for Next {prediction_days} Days")
        plt.xlabel("Date")
        plt.ylabel("Close Price")

        # Set the x-ticks to the real dates
        plt.xticks(
            predicted_dates,  # Use the real predicted dates
            [f"{date.strftime('%Y-%m-%d')}" for date in predicted_dates],  # Format the dates
            rotation=45  # Adjust this angle as needed
        )

        # Add the legend and grid
        plt.legend()
        plt.grid(True)

        # Display the plot using Streamlit
        st.pyplot(plt)



if __name__ == "__main__":
    predict_real()