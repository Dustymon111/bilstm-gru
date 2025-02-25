import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def pengujian():    
    st.markdown("<h2 style='text-align: center;'>Pengujian Model dengan K-Fold</h2>", unsafe_allow_html=True)
    st.divider()
    
    if "filtered_data" not in st.session_state and "selected_stock"  not in st.session_state:
        st.error("Data belum tersedia. Silakan pilih data terlebih dahulu.")
        return
    
    df = st.session_state["filtered_data"]  # Ambil data asli

    col15,col22 = st.columns([3,1])
    with col15:
        st.write(f"Pengujian Model dengan K-fold pada Data Saham {st.session_state['selected_stock']}. Tentukan     jumlah Lipatan K-Fold:")
    k = col22.number_input("Masukkan jumlah K (fold)", min_value=2, max_value=10, value=5, step=1)
    
    if st.button("Mulai Pengujian",use_container_width=True):
            # Normalisasi data
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])

            X, y = [], []
            window_size = 30
            close_index = 3
            for i in range(window_size, len(df_scaled)):
                X.append(df_scaled.iloc[i-window_size:i].values)
                y.append(df_scaled.iloc[i, close_index]) 
            X, y = np.array(X), np.array(y)

            tscv = TimeSeriesSplit(n_splits=k)
            mse_scores, mae_scores, rmse_scores= [], [], []

            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            # train_date_ranges = []
            # Iterate over the K-Fold splits
            for i, (train_index, test_index) in enumerate(tscv.split(X, y)):
                with st.spinner(f"Proses Pengujian Lipatan K-Fold ke-{i + 1} dari {k}"):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
            #     # Collect the date ranges for the training and testing sets
            #     train_start_date = df['Date'].iloc[train_index[0]].strftime('%Y-%m-%d')  # Start date (first entry in train_index)
            #     train_end_date = df['Date'].iloc[train_index[-1]].strftime('%Y-%m-%d')   # End date (last entry in train_index)
            #     test_start_date = df['Date'].iloc[test_index[0]].strftime('%Y-%m-%d')   # Start date (first entry in test_index)
            #     test_end_date = df['Date'].iloc[test_index[-1]].strftime('%Y-%m-%d')     # End date (last entry in test_index)
                
            #     # Calculate the lengths (sizes) of the training and testing sets
            #     train_size = len(train_index)
            #     test_size = len(test_index)
                
            #     # Add the collected date range and sizes to the list
            #     train_date_ranges.append({
            #         'Fold': i + 1,
            #         'Train Date Range': f'{train_start_date} s.d {train_end_date}',
            #         'Test Date Range': f'{test_start_date} s.d {test_end_date}',
            #         'Train Size': train_size,
            #         'Test Size': test_size
            #     })

            # # Convert the list of dictionaries into a DataFrame
            # date_ranges_df = pd.DataFrame(train_date_ranges)

            # # Display the final dataframe with all train and test date ranges and sizes
            # st.write("Train and Test Date Ranges and Sizes for All Folds:")
            # st.dataframe(date_ranges_df)

                    model = Sequential()
                    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
                    model.add(GRU(units=50, return_sequences=False))
                    model.add(Dense(1))

                    # Compile the model
                    model.compile(optimizer='adam', loss='mean_squared_error')

                    # Train the model
                    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
                
                    # Prediksi
                    predictions = model.predict(X_test)
        
                    # Evaluasi
                    mse_scores.append(mean_squared_error(y_test, predictions))
                    mae_scores.append(mean_absolute_error(y_test, predictions))
                    rmse_scores.append(np.sqrt(mean_squared_error(y_test, predictions)))
                   

            # Tabel metrik evaluasi
            metrics_df = pd.DataFrame({
                "MSE": mse_scores,
                "MAE": mae_scores,
                "RMSE": rmse_scores,
            },index=list(range(1, k + 1)))

            st.html("<h3 style='text-align:center;'>Hasil Pengujian</h3>")

            st.html("<b>Tabel Evaluasi Model</b>")
            st.dataframe(metrics_df.style.format(precision=4),use_container_width=True)

            # Rata-rata metrik
            avg_metrics = {
                "MSE": np.mean(mse_scores),
                "MAE": np.mean(mae_scores),
                "RMSE": np.mean(rmse_scores),
            }

            # Konversi JSON ke DataFrame
            df_metrics = pd.DataFrame([avg_metrics])

            # Tampilkan sebagai DataFrame
            st.write("**Rata-Rata Metrik**")
            st.dataframe(df_metrics,hide_index=True,use_container_width=True)

    col6,col7,col8= st.columns([3,1,1])
    col6.text("")
    with col7:
        if st.button("Kembali",use_container_width=True):
            st.switch_page("pages/data.py")
    with col8:
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("authenticated", None)
            st.switch_page("login.py")
            st.success("Logout berhasil! Silakan kembali ke halaman login.")

if __name__ == "__main__":
    pengujian()
