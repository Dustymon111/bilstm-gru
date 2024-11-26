import streamlit as st
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
import pymysql
from sqlalchemy import create_engine
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


db = "mysql+pymysql://root:lionel123@localhost:3306/skripsi"


def setup_sidebar():
    st.sidebar.title("Selamat Datang")
    st.sidebar.subheader("Pilih Opsi")
    option = st.sidebar.selectbox(
        'Navigasi',
        ['Data', 'Normalisasi Data', 'Pengujian','Prediksi Harga']
    )
    
    # Daftar saham (jika relevan untuk semua halaman)
    stock_options = {
        "BBCA (Bank Central Asia)": "bbca",
        "UNVR (Unilever Indonesia)": "unvr",
        "ANTM (Aneka Tambang)": "antm",
        "INDF (Indofood Sukses Makmur)": "indf"
    }
    selected_stock = st.sidebar.selectbox("Pilih Saham", options=list(stock_options.keys()))

    # Input rentang tanggal
    st.sidebar.subheader("Filter Berdasarkan Tanggal")
    start_date = st.sidebar.date_input(
        "Tanggal Mulai",
        value=date(2014, 11, 3),
        min_value=date(2014, 11, 3),
        max_value=date(2024, 10, 31)
    )
    end_date = st.sidebar.date_input(
        "Tanggal Akhir",
        value=date(2024, 10, 31),
        min_value=date(2014, 11, 3),
        max_value=date(2024, 10, 31)
    )

    # Validasi tanggal
    if start_date > end_date:
        st.sidebar.error("Tanggal Mulai tidak boleh lebih besar dari Tanggal Akhir!")
    
    return option, stock_options, selected_stock, start_date, end_date

def main():
    # Setup sidebar
    option, stock_options, selected_stock, start_date, end_date = setup_sidebar()

    # Konten utama berdasarkan opsi
    if option == 'Data':
        data(stock_options, selected_stock, start_date, end_date)
    elif option == 'Normalisasi Data':
        normalisasi(stock_options, selected_stock)
    # elif option == "Pengujian":
    #     pengujian()
    else:
        predict(stock_options, selected_stock)


def get_stock_data_from_mysql(connection, table_name):
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, connection)

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close',line=dict(color="red")))
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),  # Menambahkan range slider
            tickformat="%Y",  # Format hanya tahun
            showgrid=True,  # Menambahkan grid untuk visual yang lebih jelas
        ),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig)


def data(stock_options, selected_stock, start_date, end_date):
    st.markdown(
        """
        <h1 style="text-align: center;">Aplikasi Data Saham</h1>
        """,
        unsafe_allow_html=True
    )

    db_connection = create_engine(db)
    conn = db_connection.connect()

    # Tombol untuk menampilkan data
    if st.sidebar.button("Tampilkan Data"):
        # Nama tabel berdasarkan saham terpilih
        selected_table = stock_options[selected_stock]

        # Load CSV into a pandas DataFrame
        df = pd.read_csv(f'{selected_table.upper()}.csv', parse_dates=['Date'], index_col=['Date'])

        # Write DataFrame to MySQL
        df.sort_index().to_sql(selected_table, con=db_connection, if_exists='replace')

        # Query ke database
        query = f"""
            SELECT * 
            FROM {selected_table}
            WHERE Date BETWEEN '{start_date}' AND '{end_date}'
        """

        try:
            df = pd.read_sql(query, conn)
            if df.empty:
                st.warning("Tidak ada data yang ditemukan untuk rentang tanggal tersebut.")
            else:
                st.write(f"Data {selected_stock} dari {start_date} hingga {end_date}:")
                st.dataframe(df.style.hide(axis="index"), use_container_width=True)
                df.columns = df.columns.str.strip()
                df.set_index(df['Date'], inplace=True)
                st.line_chart(df['Close'])
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


def normalisasi(stock_options, selected_stock):
    st.markdown(
        """
        <h1 style="text-align: center;">Normalisasi Data Saham</h1>
        """,
        unsafe_allow_html=True
    )

    # Koneksi ke database
    db_connection = create_engine(db)
    conn = db_connection.connect()

    # Tombol untuk memulai proses normalisasi
    if st.button("Lakukan Normalisasi"):
        try:
            # Membaca data berdasarkan saham yang dipilih
            query = f"SELECT * FROM {stock_options[selected_stock]}"
            df = pd.read_sql(query, conn, index_col=['Date'])
            df.columns = df.columns.str.strip()
            if df.empty:
                st.warning("Tidak ada data untuk saham yang dipilih.")
            else:
                st.subheader("Data Asli")
                st.dataframe(df, use_container_width=True)

                # Inisialisasi MinMaxScaler
                scaler = MinMaxScaler()

                # Melakukan normalisasi hanya pada kolom numerik
                df_normalized = df.copy()
                df_normalized = pd.DataFrame(
                    scaler.fit_transform(df), 
                    columns=df.columns,        
                    index=df.index 
                )   

                st.subheader("Data Setelah Normalisasi")
                st.dataframe(df_normalized, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# def pengujian(stock_options, selected_stock):
#     # Judul halaman
#     st.markdown("<h2 style='text-align: center;'>Pengujian Model dengan K-Fold</h2>", unsafe_allow_html=True)

#     # Pilihan jumlah K untuk K-Fold
#     k = st.sidebar.number_input("Masukkan jumlah K (fold)", min_value=2, max_value=10, value=5, step=1)


#     # Ambil data saham dari MySQL
#     db_connection = create_engine(db)
#     conn = db_connection.connect()

#     # Normalisasi data
#     scaler = MinMaxScaler()
#     features = ['Open', 'High', 'Low', 'Close', 'Volume']  # Kolom fitur
#     df_scaled = scaler.fit_transform(df[features])

#     # Pisahkan fitur dan target
#     X = df_scaled[:, :-1]  # Semua kolom kecuali 'Close'
#     y = df_scaled[:, -1]   # Kolom 'Close'

#     # K-Fold Cross-Validation
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#     r2_scores, mse_scores, mae_scores, rmse_scores = [], [], [], []

#     actual_prices, predicted_prices = [], []  # Untuk grafik

#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Prediksi
#         predictions = model.predict(X_test)
#         predicted_prices.extend(predictions.flatten())
#         actual_prices.extend(y_test.flatten())

#         # Evaluasi
#         r2_scores.append(r2_score(y_test, predictions))
#         mse_scores.append(mean_squared_error(y_test, predictions))
#         mae_scores.append(mean_absolute_error(y_test, predictions))
#         rmse_scores.append(np.sqrt(mean_squared_error(y_test, predictions)))

#     # Tabel metrik evaluasi
#     metrics_df = pd.DataFrame({
#         "Fold": list(range(1, k + 1)),
#         "R-Squared": r2_scores,
#         "MSE": mse_scores,
#         "MAE": mae_scores,
#         "RMSE": rmse_scores
#     })

#     st.subheader("Tabel Evaluasi Model")
#     st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)

#     # Rata-rata metrik
#     avg_metrics = {
#         "R-Squared": np.mean(r2_scores),
#         "MSE": np.mean(mse_scores),
#         "MAE": np.mean(mae_scores),
#         "RMSE": np.mean(rmse_scores)
#     }

#     st.subheader("Rata-Rata Metrik")
#     st.json(avg_metrics)

#     # Grafik perbandingan aktual vs prediksi
#     st.subheader("Grafik Perbandingan Harga Aktual dan Prediksi")
#     comparison_df = pd.DataFrame({
#         "Actual": actual_prices,
#         "Predicted": predicted_prices
#     })

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(y=comparison_df["Actual"], name="Actual", line=dict(color="blue")))
#     fig.add_trace(go.Scatter(y=comparison_df["Predicted"], name="Predicted", line=dict(color="red")))
#     fig.update_layout(
#         title="Perbandingan Harga Aktual dan Prediksi",
#         xaxis_title="Index",
#         yaxis_title="Harga (Scaled)"
#     )
#     st.plotly_chart(fig)


def train(df, stock_label, X, y, window_size):
    with st.spinner('Initial Dataset Training'):
        #Splitting
        split = int(len(X) * 0.8)

        # Split the dataset
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]


        st.write(X_train.shape)
        st.write(X_test.shape)
        # Create an index for plotting
        dates = df.index[window_size:]  # Adjust dates to match the X dataset

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(dates[:split], y_train, label='Training Set', color='blue')
        plt.plot(dates[split:], y_test, label='Test Set', color='red')
        plt.axvline(x=dates[split], color='black', linestyle='--', label='Split Point')
        plt.title('Training and Test Sets from BBCA Dataset')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(plt)

        #Building Model
        model = Sequential()

        # Add the Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(GRU(units=50, return_sequences=False))

        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_absolute_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
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


def predict(stock_options, selected_stock):
    st.markdown(
        """
        <h1 style="text-align: center;">Hasil Prediksi Harga Saham</h1>
        """,
        unsafe_allow_html=True
    )

    # Koneksi ke database
    db_connection = create_engine(db)
    conn = db_connection.connect()
    X = []
    y = []

    # Input jumlah hari prediksi
    prediction_days = st.sidebar.slider("Jumlah Hari Prediksi", min_value=1, max_value=30, value=7)

    start_predict_date = st.slider("Tanggal Mulai Prediksi",
        value=date(2014, 12, 3),
        min_value=date(2014, 12, 3),
        max_value=date(2024, 10, 1),
        format="YYYY-MM-DD")
    
    calculated_start_date = start_predict_date - timedelta(days=30)
    calculated_end_date = start_predict_date + timedelta(days=prediction_days)
    

    # Membaca data terbaru dari database
    query = f"""
        SELECT * 
        FROM {stock_options[selected_stock]} 
        ORDER BY Date ASC
    """
    df_plot = pd.read_sql(query, conn, index_col=['Date'])
    df_plot.columns = df_plot.columns.str.strip()


    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot.index, df_plot['Close'], label='Dataset', color='blue')
    plt.axvline(x=start_predict_date, color='black', linestyle='--', label='Split Point')
    plt.title('Dataset Predict Start Date')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)

       # Membaca data terbaru dari database
    query = f"""
        SELECT * 
        FROM {stock_options[selected_stock]} 
        WHERE Date < '{calculated_end_date}' 
        ORDER BY Date DESC
        LIMIT {30+prediction_days} 
    """
    df = pd.read_sql(query, conn, index_col=['Date'])
    df.columns = df.columns.str.strip()

    
    # Tombol untuk memulai prediksi
    if st.button("Prediksi Harga"):
        try:

            if df.empty:
                st.warning("Data saham tidak ditemukan.")
            else:
                st.subheader("Data Terbaru")
                st.dataframe(df.style.hide(axis="index"), use_container_width=True)
                st.write(df.shape)
                st.write(df.size)

                # Preprocessing data untuk prediksi
                scaler = MinMaxScaler()
                df_scaled = scaler.fit_transform(df)

                #Windowing  
                window_size = 30
                for i in range(window_size, len(df_scaled)):
                    X.append(df_scaled[i-window_size:i])
                    y.append(df_scaled[i, df.columns.get_loc('Close')])
                X, y = np.array(X), np.array(y)

                st.write("Windowing Done")

                folder_path = os.path.join(os.getcwd(), "model")
                model_path = f"{folder_path}/{stock_options[selected_stock]}-model.h5"


                #Inisial Training
                if not os.path.exists(model_path):
                    st.write("Training dimulai")
                    train(df, stock_options[selected_stock], X, y, window_size)



                # Membentuk input untuk model (shape: [samples, timesteps, features])
                last_sequence = df_scaled[-30:]  # Data 60 hari terakhir untuk input model
                print(last_sequence[np.newaxis, :, :])

                # Memuat model
                model = load_model(model_path)  # Ganti dengan path model Anda

                # Melakukan prediksi
                predictions = []
                for _ in range(prediction_days):
                    prediction = model.predict(last_sequence[np.newaxis, :, :])
                    predictions.append(prediction[0, 0])
                    # Update sequence dengan prediksi terbaru
                    new_data = np.zeros((1, len(df.columns)))
                    new_data[0, -1] = prediction[0, 0]  # Prediksi hanya pada kolom "Close"
                    # print("new data:", new_data)
                    last_sequence = np.append(last_sequence[1:], new_data, axis=0)
                    print("last sequence:", last_sequence)

                # Denormalisasi hasil prediksi
                predictions = scaler.inverse_transform(
                    np.hstack((np.zeros((len(predictions), len(df.columns)-1)), 
                               np.array(predictions).reshape(-1, 1)))
                )[:, -1]

                # close_scaler = MinMaxScaler(feature_range=(0, 1))
                # close_prices = df[['Close']].values
                # close_scaler.fit(close_prices)
                # predictions = np.array(predictions).reshape(-1, 1)
                # original_predictions = close_scaler.inverse_transform(predictions)
                # st.write("Denormalized Predictions:", original_predictions)
                

                # Hitung tanggal prediksi ke depan
                future_dates = pd.date_range(start=df.index.min() + pd.Timedelta(days=1), periods=prediction_days)

                prediction_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": predictions
                })

                st.subheader("Hasil Prediksi")
                st.dataframe(prediction_df.style.hide(axis="index"), use_container_width=True)

                # Plot hasil prediksi
                st.subheader("Grafik Prediksi Harga Saham")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'], mode='lines', name='Data Asli'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=prediction_df['Predicted Close'], 
                    mode='lines', name='Prediksi', line=dict(color="red", dash='dash')
                ))
                st.plotly_chart(fig)

        except Exception as e:  
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()