# ใช้ image ของ Python ที่รองรับ Streamlit
FROM python:3.8-slim

# ตั้งค่า directory ใน container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt ไปยัง container
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดไปยัง container
COPY . .

# กำหนดคำสั่งที่ต้องรันเมื่อ container start ขึ้นมา
CMD ["streamlit", "run", "app.py"]
