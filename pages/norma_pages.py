import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pages import pengujian_pages



def normalize_data(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])
    return df_normalized,scaler
 
def main():
    
    st.markdown(
    "<h1 style='text-align: center;'>Normalisasi Data</h1>",
    unsafe_allow_html=True
    )

    col1,col2,col3 = st.columns([1,1,3])
    col1.html("<p style='padding-top:40px;'>Nama Perusahaan</p>")
    col2.html("<p style='padding-top:40px;'>:</p>") 
    

    if "filtered_data" in st.session_state and "selected_stock" in st.session_state:
        col3.html(f"<p style='padding-top:40px;'>{st.session_state['selected_stock']}</p>")
        
        df = st.session_state["filtered_data"]
        jumlah_data = len(df)

        col4,col5,col6 = st.columns([1,1,3])
        col4.html("<p>Jumlah Data </p>")
        col5.html("<p'>:</p>")
        col6.html(f"<p>{jumlah_data}</p>")

        # Normalisasi
        df_normalized,scaler = normalize_data(df)
        st.session_state["normalized_data"] = df_normalized  # Simpan hasil normalisasi
        st.session_state['raw_data'] = df
        st.session_state["scaler"] = scaler
         # Menampilkan tabel min dan max sesuai fitur
        min_max_df = pd.DataFrame({
            'Fitur': df.columns[1:],  # Kolom selain tanggal
            'Min': scaler.data_min_,
            'Max': scaler.data_max_
        })  
        min_max_df = min_max_df.set_index('Fitur').transpose()
        st.write("Nilai Min dan Max Setiap Fitur Sebelum Normalisasi:")
        st.dataframe(min_max_df, use_container_width=True)
        
        st.write("Data Setelah Normalisasi:")
        st.dataframe(df_normalized,use_container_width=True)
        
        col7,col8,col9= st.columns([3,1,1])
        col7.text("")
        with col8:
            if st.button("Kembali", use_container_width=True):
                st.switch_page("pages/data.py")
        with col9:
            if st.button("Selanjutnya", use_container_width=True):
                st.switch_page("pages/prediksi.py")
    else:
            st.error("Data belum tersedia. Kembali ke halaman utama dan pilih data terlebih dahulu.")

if __name__ == "__main__":
    main()
