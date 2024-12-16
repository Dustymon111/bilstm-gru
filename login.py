import streamlit as st

# Fungsi untuk login
    
st.markdown("<h1 style='text-align: center;'>Selamat Datang</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>di Aplikasi Prediksi Harga Saham Bluechip Indonesia</h3>", unsafe_allow_html=True)
st.divider()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Redirect authenticated users away from the login page
if "authenticated" in st.session_state and st.session_state.authenticated:
    st.switch_page("pages/data.py")
    st.write("Redirecting to the Home page...")
    st.rerun()

# st.title("Login Page")

# Login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")
if st.button("Login"):
    if username == "admin" and password == "password":  # Replace with real auth logic
        st.session_state.authenticated = True
        st.switch_page("pages/data.py")
        st.success("Login successful! Redirecting...")
        st.rerun()
    else:
        st.error("Invalid username or password.")