import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

# ใส่ CSS พื้นหลัง
st.markdown(
    """
    <style>
     /* ตั้งค่าพื้นหลังเป็น GIF */
    .stApp {
        background: url('https://media.giphy.com/media/QDjpIL6oNCVZ4qzGs7/giphy.gif?cid=ecf05e471al4crpkdd3sogukhtdljifp1oauch5qyap19c7w&ep=v1_gifs_search&rid=giphy.gif&ct=g');
        background-size: cover;
        background-position: center;
    }
    
    /* ปรับข้อความให้เห็นชัด */
    .centered-title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: white;
        background-color: rgba(0, 0, 0, 0.5); /* พื้นหลังโปร่งแสง */
        padding: 20px;
        border-radius: 10px;
    }
    
    /* ปรับข้อความให้มองเห็นชัด */
    .centered-text {
        text-align: center;
        font-size: 20px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5); /* พื้นหลังโปร่งแสง */
        padding: 10px;
        border-radius: 10px;
    }

    /* ปรับปุ่มให้มีพื้นหลังโปร่งแสง */
    div.stButton > button {
        background-color: rgba(52, 152, 219, 0.7); /* พื้นหลังโปร่งแสง */
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: rgba(41, 128, 185, 0.7); /* ปรับสีเมื่อ hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ใช้ Markdown + CSS class เพื่อจัดข้อความตรงกลาง
st.markdown('<h1 class="centered-title">🚀 Machine Learning & Neural Network Web App</h1>', unsafe_allow_html=True)
st.markdown('<p class="centered-text">🔹 ไปยังหน้าต่าง ๆ ได้จาก Sidebar หรือกดปุ่มด้านล่าง</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📝 Model Information"):
        st.switch_page("pages/1_Model_Information.py")
with col2:
    if st.button("📊 Model Development"):
        st.switch_page("pages/2_Model_Development.py")
with col3:
    if st.button("🤖 Machine Learning Demo"):
        st.switch_page("pages/3_Machine_Learning_Demo.py")
with col4:
    if st.button("🧠 Neural Network Demo"):
        st.switch_page("pages/4_Neural_Network_Demo.py")
