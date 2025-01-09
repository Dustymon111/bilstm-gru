import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

# Fungsi untuk login
    
st.markdown("<h1 style='text-align: center;'>Selamat Datang</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>di Aplikasi Prediksi Harga Saham Bluechip Indonesia</h3>", unsafe_allow_html=True)
st.divider()


# Database connection
db = "mysql+pymysql://root:lionel123@localhost:3306/skripsi"
db_connection = create_engine(db)
conn = db_connection.connect()

# Function to create the users table if it doesn't exist
def create_users_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    """
    conn.execute(text(create_table_query))

create_users_table()


# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Redirect authenticated users away from the login page
if st.session_state.authenticated:
    st.switch_page("pages/data.py")
    st.write("Redirecting to the Home page...")
    st.rerun()

# Login form
def login():
    db_connection = create_engine(db)
    conn = db_connection.connect()
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Fetch user data from database
        query = text("SELECT password FROM users WHERE username = :username")
        result = conn.execute(query, {"username": username}).fetchone()
        
        if result and result[0] == password:
            st.session_state.authenticated = True
            st.success("Login successful! Redirecting...")
            st.switch_page("pages/data.py")
            st.rerun()      
        else:
            st.error("Invalid username or password.")

def register():
    db_connection = create_engine(db)
    st.title("Register Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        # Validate input
        if not username or not password:
            st.error("Username and password cannot be empty!")
            return
        
        try:
            # Insert query
            query = text("""
                INSERT INTO users (username, password) VALUES (:username, :password)
            """)
            # Execute query
            with db_connection.begin() as conn:  # Ensures commit/rollback
                conn.execute(query, {"username": username, "password": password})
            st.success("Registration successful! You can now log in.")
        except IntegrityError:
            st.error("Username already exists! Please choose a different username.")
        except Exception as e:
            import traceback    
            st.error(f"Registration failed: {e}")
            st.text(traceback.format_exc())


# Main logic
page = st.selectbox("Choose an option", ["Login", "Register"])

if page == "Login":
    login()
elif page == "Register":
    register()
