import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: url('https://media.giphy.com/media/QDjpIL6oNCVZ4qzGs7/giphy.gif?cid=ecf05e471al4crpkdd3sogukhtdljifp1oauch5qyap19c7w&ep=v1_gifs_search&rid=giphy.gif&ct=g');
        background-size: cover;
        background-position: center;
    }

    .centered-title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: white;
        background-color: rgba(0, 0, 0, 0.5); /* พื้นหลังโปร่งแสง */
        padding: 20px;
        border-radius: 10px;
    }
    
    .centered-text {
        text-align: center;
        font-size: 20px;
        color: white;
        background-color: rgba(0, 0, 0, 0.5); /* พื้นหลังโปร่งแสง */
        padding: 10px;
        border-radius: 10px;
    }

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

st.sidebar.markdown(
    """
    <style>
    .sidebar-footer {
        position: absolute;
        width: 100%;
        text-align: center;
    }
    .sidebar-footer a {
        font-size: 16px;
        color: white;
        text-decoration: none;
    }
    .sidebar-footer img {
        vertical-align: middle;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <a href="https://github.com/PichchakornK/ISProject.git" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20">
            6404062663215 Pichchakorn Kongmai
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer" style="margin-top: 20px; padding: 10px; background-color: rgba(0, 0, 0, 0.5); border-radius: 10px;">
        <h3 style="color:white; text-align:center; margin-bottom: 15px;">📚 References</h3>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Dataset for ML: 
                <a href="https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv" target="_blank" style="color: #3498db; text-decoration: none;">ML Dataset Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Dataset for NN: 
                <a href="https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD" target="_blank" style="color: #3498db; text-decoration: none;">NN Dataset Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Machine Learning Tutorial: 
                <a href="https://www.youtube.com/watch?v=T2yT5vt1NaQ&list=PLoTScYm9O0GH_3VrwwnQafwWQ6ibKnEtU&index=6" target="_blank" style="color: #3498db; text-decoration: none;">ML Video Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Neural Network Tutorial: 
                <a href="https://github.com/Coding-Lane/Image-Classification-CNN-Keras.git" target="_blank" style="color: #3498db; text-decoration: none;">NN GitHub Link</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
