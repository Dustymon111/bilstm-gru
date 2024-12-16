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


def train(df, df_normalized, stock_label):
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
            y.append(df_normalized[i, df.columns.get_loc('Close')])
        X, y = np.array(X), np.array(y)

        #Splitting
        split = int(len(X) * 0.8)

        # Split the dataset
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Create an index for plotting
        dates = df.index[window_size:]  # Adjust dates to match the X dataset

        # Plotting
        # plt.figure(figsize=(12, 6))
        # plt.plot(dates[:split], y_train, label='Training Set', color='blue')
        # plt.plot(dates[split:], y_test, label='Test Set', color='red')
        # plt.axvline(x=dates[split], color='black', linestyle='--', label='Split Point')
        # plt.title('Training and Test Sets from BBCA Dataset')
        # plt.xlabel('Date')
        # plt.ylabel('Close Price')
        # plt.legend()
        # st.pyplot(plt)

        #Building Model
        model = Sequential()

        # Add the Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(GRU(units=50, return_sequences=False))

        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_absolute_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        st.write("Fitting Model")
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # Create a separate scaler for y (Close values)
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices = df[['Close']].values
        close_scaler.fit(close_prices)  # Fit on all 'Close' values

        testing = model.predict(X_test)

        # Inverse-transform predictions and y_test
        predicted_stock_price = close_scaler.inverse_transform(testing.reshape(-1, 1))
        y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

        
        test_dates = df.index[len(X_train) + window_size:len(X_train) + window_size + len(y_test)]
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(test_dates, y_test_actual, color='blue', label='Actual Stock Price')
        plt.plot(test_dates, predicted_stock_price, color='red', label='Predicted Stock Price')
        plt.title("Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()

        # Render the plot in Streamlit
        st.pyplot(plt)

        folder_path = os.path.join(os.getcwd(), "model")
        model.save(f'{folder_path}/{stock_label}-model.h5')
  
    st.success("Training Complete!")


def predict():
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
    st.divider()
    st.text(f"Prediksi harga Saham {st.session_state['selected_stock']}, Tentukan jumlah hari dan Tanggal mulai prediksi : ")
    db = "mysql+pymysql://root:lionel123@localhost:3306/skripsi"
    # Koneksi ke database
    db_connection = create_engine(db)   
    conn = db_connection.connect()

    col1,col2 = st.columns(2)
    # Input jumlah hari prediksi
    prediction_days = col1.slider("Jumlah Hari Prediksi", min_value=1, max_value=30, value=7)

    # Query the dates from your SQL table
    query = f"SELECT DISTINCT Date FROM {selected_table} ORDER BY Date ASC"
    dates_df = pd.read_sql(query, conn)

    # Ensure the Date column is of datetime type
    dates_df['Date'] = pd.to_datetime(dates_df['Date'])

    # Convert to a list of unique dates
    valid_dates = dates_df['Date'].dt.date.tolist()

    # Create a select box for weekdays
    start_predict_date = col2.selectbox(
        "Tanggal Mulai Prediksi",
        options=valid_dates[30:-29],
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )

    # start_predict_date = st.slider("Tanggal Mulai Prediksi",
    #     min_value=weekdays[0],
    #     max_value=weekdays[-1],
    #     value=weekdays[0],
    #     format="YYYY-MM-DD")

    
    # Get the index of the selected start_predict_date in valid_dates
    start_index = valid_dates.index(start_predict_date)

    # Get the next N valid dates starting from start_predict_date
    predicted_dates = valid_dates[start_index:start_index + prediction_days]

    calculated_end_date = valid_dates[start_index + prediction_days - 1]
    # st.write(calculated_end_date)
    
    # Membaca data terbaru dari database
    query = f"""
        SELECT * 
        FROM {selected_table} 
        ORDER BY Date ASC
    """
    df_all = pd.read_sql(query, conn, index_col=['Date'])
    df_all.columns = df_all.columns.str.strip()


    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df_all.index, df_all['Close'], label='Dataset', color='blue')
    plt.axvline(x=start_predict_date, color='black', linestyle='--', label='Split Point')
    plt.title('Dataset Predict Start Date')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)

    # Membaca data dari database
    query = f"""
        SELECT * 
        FROM {selected_table}
        WHERE Date <= '{calculated_end_date}' 
        ORDER BY Date DESC
        LIMIT {prediction_days+30} 
    """
    df = pd.read_sql(query, conn, index_col=['Date'])
    df.columns = df.columns.str.strip()
    df.sort_index(inplace=True)

    
    # Tombol untuk memulai prediksi
    if st.button("Prediksi Harga",use_container_width=True):
        try:
            if df.empty:
                st.warning("Data saham tidak ditemukan.")
            else:
                # Preprocessing data untuk prediksi
                scaler = MinMaxScaler()
                # st.dataframe(df)
                df_scaled = scaler.fit_transform(df)
                df_all_scaled = scaler.fit_transform(df_all)
                # st.dataframe(df_scaled)

                #Windowing
                X_pred = []
                y_pred = []
                window_size = 30
                for i in range(window_size, len(df_scaled)):
                    X_pred.append(df_scaled[i-window_size:i])
                    y_pred.append(df_scaled[i, df.columns.get_loc('Close')])
                X_pred, y_pred = np.array(X_pred), np.array(y_pred)


                folder_path = os.path.join(os.getcwd(), "model")
                model_path = f"{folder_path}/{selected_table}-model.h5"

                #Inisial Training
                if not os.path.exists(model_path):
                    train(df_all, df_all_scaled, selected_table)

                # Memuat model
                model = load_model(model_path)  # Ganti dengan path model Anda

                # Melakukan prediksi
                prediction = model.predict(X_pred)

                close_scaler = MinMaxScaler(feature_range=(0, 1))
                close_prices = df[['Close']].values
                close_scaler.fit(close_prices)


                y_reshaped = y_pred.reshape(-1, 1)

                # Inverse transform both `prediction` and `y`
                original_predictions = close_scaler.inverse_transform(prediction).reshape(-1, 1)
                original_actual = close_scaler.inverse_transform(y_reshaped).reshape(-1, 1)
                
               
                # Hitung tanggal prediksi ke depan
                prediction_df = pd.DataFrame({
                    "Date": predicted_dates,  # Ensure `dates` matches the length of predictions and actuals
                    "Prediction": original_predictions.flatten(),
                    "Actual": original_actual.flatten()
                }).set_index("Date")

                prediction_df = prediction_df.astype(int)

                st.html("<h3 style='text-align:center;'>Hasil Prediksi</h3>")

                # Plot hasil prediksi
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'], mode='lines', name='Data Asli'
                ))
                fig.add_trace(go.Scatter(
                    x=prediction_df.index, y=prediction_df['Prediction'],
                    mode='lines', name='Prediksi', line=dict(color="red")
                ))
                fig.update_layout(
                    title="Grafik Prediksi Harga Saham",
                    xaxis_title="Date",
                    yaxis_title="Harga "
                    )
                st.plotly_chart(fig)

                st.html("<b>Hasil Prediksi</b>")
                st.dataframe(prediction_df.style.hide(axis="index"), use_container_width=True)  

                mse_scores= mean_squared_error(original_actual, original_predictions)
                mae_scores= mean_absolute_error(original_actual, original_predictions)
                rmse_scores= np.sqrt(mean_squared_error(original_actual, original_predictions))
                mape_scores = np.mean(np.abs((original_actual - original_predictions) / original_actual)) * 100  # MAPE in percentage

                metrics = {
                "MSE": mse_scores,
                "MAE": mae_scores,
                "RMSE": rmse_scores,
                "MAPE": mape_scores
                }

                df_metrics = pd.DataFrame([metrics])

                # Tampilkan sebagai DataFrame
                st.write("**Metrik Prediksi**")
                st.dataframe(df_metrics,hide_index=True,use_container_width=True)

                col7,col8,col9= st.columns([3,1,1])
                col7.text("")
                with col8:
                    if st.button("Kembali", use_container_width=True):
                        st.switch_page("pages/norma_pages.py")
                with col9:
                    if st.button("Ubah data", use_container_width=True):
                        st.switch_page("../data.py")
        except Exception as e:  
            st.error(f"Terjadi kesalahan: {e}")


if __name__ == "__main__":
    predict()