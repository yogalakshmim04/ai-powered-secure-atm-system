# 🏦 AI-Powered Secure ATM Transaction System
### Using Face Recognition and Hybrid PIN Technology

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange)

---

## 📌 About the Project

A next-generation software-based ATM simulation system that replaces traditional card-and-PIN authentication with:

- 🔍 **LBPH Face Recognition** — Biometric identity verification via webcam
- 🔢 **Hybrid PIN Technology** — Optical illusion PIN pad that prevents shoulder surfing
- 🪪 **Aadhaar KYC Verification** — Simulated identity verification during registration
- 💰 **Complete Banking Dashboard** — Deposit, Withdraw, Balance, History

> Operates **fully offline** — no cloud dependency, no special hardware required.

---

## 🚀 Features

- ✅ Cardless authentication using face recognition
- ✅ Hybrid PIN pad — real digit visible up close, fake digit visible from distance
- ✅ Aadhaar KYC with OTP verification during registration
- ✅ 8-sample face data augmentation for stronger recognition
- ✅ SHA-256 PIN hashing — raw PIN never stored
- ✅ 3-attempt face recognition limit with account lockout
- ✅ Complete banking operations — deposit, withdraw, balance, history
- ✅ PDF account summary download on registration
- ✅ Dark / Light theme toggle
- ✅ Responsive professional UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Framework | Flask |
| Database | MySQL 8.0 |
| Face Detection | Haar Cascade Classifier (OpenCV) |
| Face Recognition | LBPH Algorithm (OpenCV) |
| Image Processing | OpenCV, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Webcam API | WebRTC getUserMedia |
| Security | SHA-256 (hashlib) |
| PDF Generation | html2canvas, jsPDF |

---

## 📁 Project Structure

atm_face_recognition/
│
├── __pycache__/
│   └── config.cpython-312.pyc
│
├── face_module/
│   ├── __pycache__/
│   ├── capture.py
│   ├── recognize.py
│   └── train.py
│
├── models/
│   └── lbph_model.yml
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── img/
│   │   └── favicon.svg
│   │
│   └── js/
│       └── pin_shuffle.js
│
├── templates/
│   ├── aadhaar.html
│   ├── atm_menu.html
│   ├── landing.html
│   ├── login.html
│   ├── pin.html
│   ├── register.html
│   └── success.html
│
├── uploads/
│
├── app.py
├── config.py
├── generate_hybrid_pins.py
├── readme.md
└── requirements.txt

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/atm-face-recognition.git
cd atm-face-recognition
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — MySQL Setup
```sql
SOURCE database.sql;
```

### Step 4 — Configure Database
Edit `config.py`:
```python
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password",
        database="atm_face_db"
    )
```

### Step 5 — Create Folders
```bash
mkdir models
mkdir uploads
```

### Step 6 — Run
```bash
python app.py
```

### Step 7 — Open Browser
http://localhost:5000


---

## 🔄 Usage Flow

### Registration

Fill Personal Details → Face Capture → Set PIN → Aadhaar KYC → OTP Verify → Account Created


### Login
Hybrid PIN Entry → SHA-256 Verify → Face Scan → LBPH Predict → Dashboard Access



---

## 🔐 Security Features

| Feature | Implementation |
|---------|---------------|
| PIN Storage | SHA-256 Cryptographic Hash |
| Face Data | Local Filesystem Only |
| Session | Flask Server-Side Sessions |
| Account Lockout | 3 Failed Face Attempts |
| Aadhaar | Masked Display (XXXX XXXX 1234) |

---

## 🖥️ System Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10 / Ubuntu 20.04 |
| Python | 3.8+ |
| RAM | 4 GB |
| Webcam | 720p or higher |
| Browser | Chrome / Firefox / Edge |

---

## 🔮 Future Enhancements

-  Liveness detection to prevent photo spoofing
-  Fingerprint authentication
-  Real banking API integration
-  Raspberry Pi deployment

---

## 👩‍💻 Developed By

**Yoga Lakshmi**
Department of Computer Science
Tamil Nadu, India

