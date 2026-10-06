# app.py — NexVault Bank ATM
# Flow: Register(details+face+pin) → Login(PIN→Face→Dashboard)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json as json_mod
from config import get_db
import hashlib, base64, os, numpy as np, cv2

app = Flask(__name__)
app.secret_key = "faceatm_secret_2024"

UPLOAD_FOLDER = "uploads"
MODEL_PATH    = "models/lbph_model.yml"

# ─── Helpers ────────────────────────────────────────────

def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

def b64_to_gray_array(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

def detect_and_crop_face(gray_img):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = detector.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(gray_img[y:y+h, x:x+w], (200, 200))

def augment_face(face_img):
    samples = [face_img]
    samples.append(cv2.flip(face_img, 1))
    h, w = face_img.shape
    cx, cy = w // 2, h // 2
    for angle in [-10, 10, -5, 5]:
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(face_img, M, (w, h))
        samples.append(rot)
    bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=20)
    dark   = cv2.convertScaleAbs(face_img, alpha=0.8, beta=-20)
    samples.append(bright)
    samples.append(dark)
    return samples

def save_face_to_disk(b64_str, user_id):
    gray = b64_to_gray_array(b64_str)
    face = detect_and_crop_face(gray)
    if face is None:
        return None
    folder = os.path.join(UPLOAD_FOLDER, str(user_id))
    os.makedirs(folder, exist_ok=True)
    samples = augment_face(face)
    for idx, sample in enumerate(samples):
        path = os.path.join(folder, f'face_{idx}.jpg')
        cv2.imwrite(path, sample)
    return folder

def train_all_models():
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8
    )
    faces, labels = [], []
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, face_path FROM users WHERE face_trained=TRUE AND face_path IS NOT NULL"
    )
    rows = cursor.fetchall()
    cursor.close(); db.close()
    for row in rows:
        folder = row['face_path']
        if not folder or not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.startswith('face_') or not fname.endswith('.jpg'):
                continue
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, (200, 200)))
                labels.append(row['id'])
    if len(faces) >= 2:
        os.makedirs('models', exist_ok=True)
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        return True
    return False

def verify_face_against_user(b64_image, user_id):
    if not os.path.exists(MODEL_PATH):
        return False
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.read(MODEL_PATH)
    gray = b64_to_gray_array(b64_image)
    face = detect_and_crop_face(gray)
    if face is None:
        return False
    label, confidence = recognizer.predict(face)
    print(f"[FACE] predicted_label={label}, user_id={user_id}, confidence={confidence:.2f}")

    if label != user_id:
        print(f"[FACE] REJECTED — label mismatch ({label} != {user_id})")
        return False

    # RELAXED threshold — 85 works for normal lighting
    if confidence >= 85:
        print(f"[FACE] REJECTED — confidence too high ({confidence:.2f} >= 85)")
        return False

    folder = os.path.join(UPLOAD_FOLDER, str(user_id))
    if os.path.isdir(folder):
        stored_faces = []
        for fname in os.listdir(folder):
            if fname.startswith('face_') and fname.endswith('.jpg'):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    stored_faces.append(cv2.resize(img, (200, 200)))
        if stored_faces:
            hist_in = cv2.calcHist([face], [0], None, [256], [0,256])
            cv2.normalize(hist_in, hist_in)
            similarities = []
            for sf in stored_faces:
                hist_sf = cv2.calcHist([sf], [0], None, [256], [0,256])
                cv2.normalize(hist_sf, hist_sf)
                score = cv2.compareHist(hist_in, hist_sf, cv2.HISTCMP_CORREL)
                similarities.append(score)
            best_sim = max(similarities)
            print(f"[FACE] histogram similarity = {best_sim:.4f}")
            # RELAXED threshold — 0.35
            if best_sim < 0.35:
                print(f"[FACE] REJECTED — histogram too different ({best_sim:.4f} < 0.35)")
                return False

    print(f"[FACE] ACCEPTED — label={label}, conf={confidence:.2f}")
    return True

# ─── Routes ─────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')

# ── REGISTER ──────────────────────────────────────────

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/register/aadhaar', methods=['GET'])
def register_aadhaar():
    return render_template('aadhaar.html')

@app.route('/register/submit', methods=['POST'])
def register_submit():
    try:
        data = request.get_json()
        firstname      = data.get('firstname', '').strip()
        lastname       = data.get('lastname', '').strip()
        dob            = data.get('dob', '').strip()
        age            = int(data.get('age', 0))
        phone          = data.get('phone', '').strip()
        address        = data.get('address', '').strip()
        city           = data.get('city', '').strip()
        state          = data.get('state', '').strip()
        pincode        = data.get('pincode', '').strip()
        account        = data.get('account_number', '').strip()
        balance        = float(data.get('balance', 0) or 0)
        pin            = data.get('pin', '')
        face_image     = data.get('face_image', '')
        aadhar_number  = data.get('aadhar_number', '').strip()

        import re
        if not re.match(r'^[a-zA-Z\s]+$', firstname) or len(firstname) < 2:
            return jsonify(success=False, error='Invalid first name.')
        if not re.match(r'^[a-zA-Z\s]+$', lastname) or len(lastname) < 2:
            return jsonify(success=False, error='Invalid last name.')
        if not dob:
            return jsonify(success=False, error='Date of birth required.')
        if age < 18:
            return jsonify(success=False, error='Must be 18 years or older.')
        if not re.match(r'^\+91[6789]\d{9}$', phone):
            return jsonify(success=False, error='Invalid phone number.')
        if not all([address, city, state]):
            return jsonify(success=False, error='Complete address required.')
        if not re.match(r'^\d{6}$', pincode):
            return jsonify(success=False, error='Invalid pincode.')
        if not account:
            return jsonify(success=False, error='Account number required.')
        if balance < 500:
            return jsonify(success=False, error='Minimum initial balance is ₹500.')
        if not re.match(r'^\d{4}$', pin):
            return jsonify(success=False, error='PIN must be exactly 4 digits.')
        if not re.match(r'^\d{12}$', aadhar_number):
            return jsonify(success=False, error='Aadhaar must be 12 digits.')
        if not face_image:
            return jsonify(success=False, error='Face image required.')

        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE account_number=%s", (account,))
        if cursor.fetchone():
            cursor.close(); db.close()
            return jsonify(success=False, error='Account number already exists.')

        gray       = b64_to_gray_array(face_image)
        face_check = detect_and_crop_face(gray)
        if face_check is None:
            cursor.close(); db.close()
            return jsonify(success=False, error='No face detected. Please retake photo.')

        cursor.execute(
            "INSERT INTO users "
            "(firstname, lastname, dob, age, account_number, phone, address, city, state, pincode, aadhar_number, pin, balance) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (firstname, lastname, dob, age, account, phone,
             address, city, state, pincode, aadhar_number,
             hash_pin(pin), balance)
        )
        db.commit()
        user_id = cursor.lastrowid

        face_folder = save_face_to_disk(face_image, user_id)
        if face_folder is None:
            cursor.close(); db.close()
            return jsonify(success=False, error='Face save failed. Please retake in good lighting.')

        cursor.execute(
            "UPDATE users SET face_path=%s, face_trained=TRUE WHERE id=%s",
            (face_folder, user_id)
        )
        db.commit()
        cursor.close(); db.close()
        train_all_models()
        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/register/success')
def register_success():
    return render_template('success.html')

# ── LOGIN ──────────────────────────────────────────────

@app.route('/login', methods=['GET'])
def login():
    import cv2, numpy as np, random, base64
    registered = request.args.get('registered')
    session.clear()

    def get_num_canvas(text, size=200):
        canvas = np.ones((size, size), dtype="uint8") * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, str(text), (40, 160), font, 5, (0, 0, 0), 18, cv2.LINE_AA)
        return canvas

    def create_illusion_btn(real_num, fake_num):
        img_real  = get_num_canvas(real_num)
        img_fake  = get_num_canvas(fake_num)
        far_view  = cv2.GaussianBlur(img_fake, (35, 35), 12)
        real_blur = cv2.GaussianBlur(img_real, (35, 35), 3)
        near_view = cv2.subtract(img_real, real_blur)
        hybrid    = cv2.add(far_view, near_view)
        hybrid    = cv2.normalize(hybrid, None, 0, 255, cv2.NORM_MINMAX)
        return hybrid

    # real_order FIXED 0-9 — includes 0
    real_order = list(range(0, 10))
    fake_order = list(range(0, 10))
    random.shuffle(fake_order)
    for i in range(10):
        if fake_order[i] == real_order[i]:
            swap = (i+1) % 10
            fake_order[i], fake_order[swap] = fake_order[swap], fake_order[i]

    btns = []
    for i in range(10):
        btns.append(create_illusion_btn(real_order[i], fake_order[i]))

    # 3x4 grid — 0-8 in 3x3, then empty|9|empty
    empty = np.ones((200, 200), dtype="uint8") * 255
    rows = []
    for i in range(0, 9, 3):
        rows.append(np.hstack([btns[i], btns[i+1], btns[i+2]]))
    rows.append(np.hstack([empty, btns[9], empty]))
    final_board = np.vstack(rows)

    _, buf = cv2.imencode('.png', final_board)
    pin_img = base64.b64encode(buf).decode('utf-8')

    session['pin_real_order'] = real_order

    return render_template('login.html', pin_img=pin_img, real_order=real_order, registered=registered)

@app.route('/login/open-pin', methods=['POST'])
def open_pin_window():
    import cv2, numpy as np, random

    def get_num_canvas(text, size=200, color=(0,0,0), thickness=15):
        canvas = np.ones((size, size), dtype="uint8") * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, str(text), (45, 155), font, 4.5, color, thickness, cv2.LINE_AA)
        return canvas

    def create_pro_illusion(real_n, fake_n):
        img_fake     = get_num_canvas(fake_n)
        shadow       = cv2.GaussianBlur(img_fake, (45, 45), 15)
        img_real     = get_num_canvas(real_n)
        white_mask   = get_num_canvas(real_n, thickness=25)
        white_border = cv2.GaussianBlur(white_mask, (11, 11), 5)
        real_blur    = cv2.GaussianBlur(img_real, (25, 25), 3)
        near_view    = cv2.subtract(img_real, real_blur)
        temp         = cv2.addWeighted(shadow, 0.5, white_border, 0.5, 0)
        final        = cv2.add(temp, near_view)
        return cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)

    entered_pin = [""]
    real_order = list(range(10))
    random.shuffle(real_order)
    fake_order = list(range(10))
    random.shuffle(fake_order)

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            col, row = x // 200, y // 200
            idx = row * 3 + col
            if idx < len(real_order):
                clicked = real_order[idx]
                if len(entered_pin[0]) < 4:
                    entered_pin[0] += str(clicked)
                    print(f"Typed PIN: {'*' * len(entered_pin[0])}")

    btns  = [create_pro_illusion(real_order[i], fake_order[i]) for i in range(10)]
    empty = np.ones((200, 200), dtype="uint8") * 255
    rows  = []
    for i in range(0, 9, 3):
        rows.append(np.hstack([btns[i], btns[i+1], btns[i+2]]))
    rows.append(np.hstack([empty, btns[9], empty]))
    final_board = np.vstack(rows)

    cv2.namedWindow('Illusion PIN Pad')
    cv2.setMouseCallback('Illusion PIN Pad', mouse_click)

    while True:
        cv2.imshow('Illusion PIN Pad', final_board)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(entered_pin[0]) == 4:
            break
        elif key == 8:
            entered_pin[0] = entered_pin[0][:-1]
        elif key == 27:
            entered_pin[0] = ""
            break

    cv2.destroyAllWindows()

    pin = entered_pin[0]
    if not pin:
        return jsonify({'ok': False, 'error': 'cancelled'})

    db     = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, firstname, lastname FROM users WHERE pin=%s AND face_trained=TRUE",
                   (hash_pin(pin),))
    user = cursor.fetchone()
    cursor.close(); db.close()

    if user:
        session['pending_user_id']   = user['id']
        session['pending_user_name'] = user['firstname'] + ' ' + user['lastname']
        session['face_attempts']     = 0
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'wrong_pin'})

@app.route('/login/verify-pin', methods=['POST'])
def verify_pin():
    pin = request.form.get('pin', '')
    db  = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, firstname, lastname FROM users WHERE pin=%s AND face_trained=TRUE",
                   (hash_pin(pin),))
    user = cursor.fetchone()
    cursor.close(); db.close()
    if user:
        session['pending_user_id']   = user['id']
        session['pending_user_name'] = user['firstname'] + ' ' + user['lastname']
        session['face_attempts']     = 0
        return jsonify({'ok': True})
    return jsonify({'ok': False})

# ── LOGIN — Step 2: Face ───────────────────────────────

@app.route('/login/face', methods=['POST'])
def login_face():
    if 'pending_user_id' not in session:
        return jsonify({'ok': False, 'error': 'Session expired'})

    user_id    = session['pending_user_id']
    face_image = request.form.get('face_image', '')
    attempts   = session.get('face_attempts', 0)

    if attempts >= 3:
        session.clear()
        return jsonify({'ok': False, 'blocked': True})

    match = verify_face_against_user(face_image, user_id)
    session['face_attempts'] = attempts + 1

    if match:
        # Get name BEFORE popping from session
        user_name = session.get('pending_user_name', '')
        session['user_id']  = user_id
        session['verified'] = True
        session.pop('pending_user_id',   None)
        session.pop('pending_user_name', None)
        session.pop('face_attempts',     None)
        return jsonify({'ok': True, 'name': user_name})
    else:
        remaining = 3 - session['face_attempts']
        if remaining <= 0:
            session.clear()
            return jsonify({'ok': False, 'blocked': True, 'remaining': 0})
        return jsonify({'ok': False, 'remaining': remaining})

# ── DASHBOARD ──────────────────────────────────────────

@app.route('/atm')
def atm():
    if not session.get('verified'):
        return redirect(url_for('login'))
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id=%s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close(); db.close()
    if not user:
        return redirect(url_for('login'))
    return render_template('atm_menu.html', user=user)

# ── TRANSACTION ────────────────────────────────────────

@app.route('/transaction', methods=['POST'])
def transaction():
    if not session.get('verified'):
        return jsonify({'error': 'Unauthorized'}), 401

    t_type = request.form.get('type')
    amount = float(request.form.get('amount', 0))

    if t_type not in ('DEPOSIT', 'WITHDRAW') or amount <= 0:
        return jsonify({'error': 'Invalid request'}), 400

    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT balance FROM users WHERE id=%s", (session['user_id'],))
    user    = cursor.fetchone()
    balance = float(user['balance'])

    if t_type == 'WITHDRAW':
        if amount > balance:
            cursor.close(); db.close()
            return jsonify({'error': 'Insufficient balance!'}), 400
        balance -= amount
    else:
        balance += amount

    cursor.execute("UPDATE users SET balance=%s WHERE id=%s", (balance, session['user_id']))
    cursor.execute(
        "INSERT INTO transactions (user_id, type, amount) VALUES (%s, %s, %s)",
        (session['user_id'], t_type, amount)
    )
    db.commit()
    cursor.close(); db.close()
    return jsonify({'balance': balance, 'message': f'{t_type} successful!'})

# ── HISTORY ────────────────────────────────────────────

@app.route('/transactions/history')
def tx_history():
    if not session.get('verified'):
        return jsonify([])
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT type, amount, timestamp FROM transactions "
        "WHERE user_id=%s ORDER BY timestamp DESC LIMIT 20",
        (session['user_id'],)
    )
    rows = cursor.fetchall()
    cursor.close(); db.close()
    for r in rows:
        r['timestamp'] = str(r['timestamp'])
    return jsonify(rows)

# ── LOGOUT ─────────────────────────────────────────────

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

# ── RUN ────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)

