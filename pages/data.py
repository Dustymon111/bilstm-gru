import streamlit as st
import pandas as pd
from datetime import date
# import pymysql
from sqlalchemy import create_engine, inspect
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



st.set_page_config(initial_sidebar_state="collapsed", menu_items=None)

db = "mysql+pymysql://root:lionel123@localhost:3306/skripsi"

def show_pages_button():
    col10,col11 = st.columns(2)
    with col10:
        if st.button("Pengujian K-Fold", use_container_width=True):
            st.switch_page("pages/pengujian_pages.py")
    with col11:
        if st.button("Pengujian Prediksi", use_container_width=True):
            st.switch_page("pages/prediksi_pengujian.py")
    if st.button("Prediksi", use_container_width=True):
        st.switch_page("pages/norma_pages.py")

def visualize_data(df):
    df['Date'] = df['Date'].dt.date
    df.set_index(df['Date'], inplace=True)
    df.drop(columns=['Date'], inplace=True, errors='ignore')
    st.dataframe(df.style.hide(axis="index"), use_container_width=True)
    df.columns = df.columns.str.strip()
    st.line_chart(df['Close'], x_label="Date", y_label="Price (Close)")
                
# Menyediakan tombol download untuk file CSV
@st.cache_data
def convert_df(df):
    # Mengonversi DataFrame menjadi CSV dalam format byte
    return df.to_csv(index=False).encode('utf-8')

def get_stock_data_from_mysql(connection, table_name):
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, connection)

def data():
    stock_options = {
        "BBCA (Bank Central Asia)": "bbca",
        "UNVR (Unilever Indonesia)": "unvr",
        "ANTM (Aneka Tambang)": "antm",
        "INDF (Indofood Sukses Makmur)": "indf"
    }
    return stock_options


def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.switch_page("login.py")
    else:
        st.markdown(
            "<h1 style='text-align: center;'>Input Data Saham</h1>",
            unsafe_allow_html=True
        )
        db_connection = create_engine(db)
        # Tombol Logout
        col00,col11= st.columns([6,1])
        col00.text("")
        if col11.button("Logout"):
            st.session_state.pop("authenticated", None)
        method = st.selectbox("", placeholder="Pilih perusahaan", options=["Saham Aplikasi","Upload CSV"])
        
        if method == "Saham Aplikasi":
            stock_options = data()
            getdata(stock_options, db_connection)
        else:
            upload_file(db_connection)


def getdata (stock_options, db_connection):
    if "selected_index" not in st.session_state:
        st.session_state['selected_index'] = None
    

    stock_list = list(stock_options.keys()) 

    db_connection = create_engine(db)
    conn = db_connection.connect()

     
    col1,col2,col3 = st.columns([1,1,3])
    col1.html("<p style='padding-top:40px;'>Nama Perusahaan</p>")
    col2.html("<p style='padding-top:40px;'>:</p>")
    selected_stock = col3.selectbox("",index=st.session_state["selected_index"] ,placeholder="Pilih perusahaan", options=stock_list)
    
    col4,col5,col6 = st.columns([1,1,3])
    col4.html("<p>File data Saham </p>")
    col5.html("<p'>:</p>")
    
    if selected_stock:
        col7,col8,col9 = st.columns([1,1,4])
        with col7:
            start_date = st.date_input(
                "Tanggal Mulai",
                value=date(2014, 11, 3),
                min_value=date(2014, 11, 3),
                max_value=date(2024, 10, 31)
                )
        with col8:
            end_date = st.date_input(
                "Tanggal Akhir",
                value=date(2024, 10, 31),
                min_value=date(2014, 11, 3),
                max_value=date(2024, 10, 31)
            )
        
        col9.text("")
        # Validasi tanggal
        if start_date > end_date:
            st.warning("Tanggal Mulai tidak boleh lebih besar dari Tanggal Akhir!")
            
        # Nama tabel berdasarkan saham terpilih
        selected_table = stock_options[selected_stock]
        
        # Load CSV into a pandas DataFrame
        df = pd.read_csv(f'{selected_table.upper()}.csv', parse_dates=['Date'], index_col=['Date'])

        # Write DataFrame to MySQL
        df.sort_index().to_sql(selected_table, con=db_connection, if_exists='replace')
        # Query ke database untuk mengambil data saham berdasarkan rentang tanggal
        query = f"""
            SELECT * 
            FROM {selected_table}
            WHERE Date BETWEEN '{start_date}' AND '{end_date}'
        """ 
        try:                
            # Mengambil data dari database
            df = pd.read_sql(query, conn)
            
            if df.empty:
                st.warning("Tidak ada data yang ditemukan untuk rentang tanggal tersebut.")
            else:
                # Menampilkan data
                st.write(f"Data {selected_stock} dari {start_date} hingga {end_date}:")
                visualize_data(df)
           
                st.session_state["selected_stock"] = selected_stock
                st.session_state["selected_table"] = selected_table
                st.session_state["filtered_data"] = df
                st.session_state["selected_index"] = stock_list.index(selected_stock)
                
                # Tombol unduh file CSV
                csv_data = convert_df(df)
                with col6:
                    st.download_button(
                        label= f"Unduh Data {selected_stock}.CSV",
                        data=csv_data,
                        file_name=f"{selected_table}_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                show_pages_button()
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.html("<p style='text-align: center; color: #8e8e8e; padding-top:40px'>Pilih Perusahaan saham terlebih dahulu...</p>")

def upload_file(db_connection):
    st.write("File yang di-upload harus memiliki 5 kolom dengan urutan seperti di bawah:")
    st.write("Date, Open, High, Low, Close, Volume")
    column_order = ["Open", "High", "Low", "Close", "Volume"]
    file_uploader = st.file_uploader("Upload File CSV", type=["csv"])
    valid_data = False
    if file_uploader:
        data = pd.read_csv(file_uploader, parse_dates=["Date"], index_col=['Date'])
        data.columns = data.columns.str.strip()

        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(data.columns.str.strip()) or list(data.columns.str.strip()) != column_order:
            st.error("File yang di-upload tidak memenuhi ketentuan seperti di atas.")  
        else:
            filename = '.'.join(file_uploader.name.split('.')[:-1]).lower()
            # Save the DataFrame to the database with the unique table name
            data.sort_index().to_sql(filename, con=db_connection, if_exists='replace')
            data = data.sort_index(ascending=True)
            st.dataframe(data.style.hide(axis="index"), use_container_width=True)       
            
            # Clean up the data
            st.line_chart(data['Close'], x_label="Date", y_label="Price (Close)")

            # Store selected stock and filtered data in session state
            st.session_state["selected_stock"] = filename
            st.session_state["selected_table"] = filename
            st.session_state["filtered_data"] = data
            valid_data = True

    if valid_data:
        show_pages_button()


if __name__ == "__main__":
    main()