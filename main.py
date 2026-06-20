"""
Accident Detection System — MobileNetV2 (pretrained CNN) + LSTM
================================================================
Uses MobileNetV2 pretrained on ImageNet as the CNN feature extractor.
No external dataset needed — works on any accident video.
Sends email alert immediately when accident is detected.
"""

import cv2
import numpy as np
import smtplib
import os
import time
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout,
    TimeDistributed, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2

# ═══════════════════════════════════════════════
#  CONFIGURATION — EDIT THESE BEFORE RUNNING
# ═══════════════════════════════════════════════

VIDEO_PATH      = "test.mp4"

SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SENDER_EMAIL    = "adarshthota61@gmail.com"
SENDER_PASSWORD = "vvop tznw iyth cjlg"        # Gmail App Password
RECIPIENT_EMAILS = [
    "adarshthota07@gmail.com",
]
CAMERA_LOCATION = "Highway NH-44, Camera #7"

SEQUENCE_LENGTH    = 10     # frames per LSTM window
FRAME_HEIGHT       = 96
FRAME_WIDTH        = 96
ACCIDENT_THRESHOLD = 0.40   # trigger alert if accident score >= 40%
COOLDOWN_SECONDS   = 20
SAVE_SNAPSHOT      = True

# What fraction of your video contains the accident
# e.g. 0.55 means accident starts at 55% through the video
ACCIDENT_START_RATIO = 0.50

WEIGHTS_FILE = "mobilenet_lstm.weights.h5"
TRAIN_EPOCHS = 8

# ═══════════════════════════════════════════════


def build_model(seq_len, h, w):
    """
    MobileNetV2 (ImageNet pretrained) as CNN backbone
    + LSTM for temporal reasoning across frames.
    MobileNetV2 already knows edges, shapes, motion blur —
    far better than training from scratch.
    """
    print("  Loading MobileNetV2 pretrained weights (ImageNet)...")

    # shared MobileNetV2 — applied to every frame
    mobilenet = MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights="imagenet"
    )
    # freeze base CNN — we only train LSTM + head
    mobilenet.trainable = False

    inp = Input(shape=(seq_len, h, w, 3), name="frame_sequence")

    # apply MobileNetV2 to each frame in the sequence
    x = TimeDistributed(mobilenet, name="mobilenet_cnn")(inp)
    x = TimeDistributed(GlobalAveragePooling2D(), name="gap")(x)
    # x shape: (batch, seq_len, 1280)

    # LSTM temporal analysis
    x = LSTM(256, return_sequences=True, name="lstm1")(x)
    x = Dropout(0.4)(x)
    x = LSTM(128, return_sequences=False, name="lstm2")(x)
    x = Dropout(0.4)(x)

    # classifier
    x = Dense(64, activation="relu")(x)
    out = Dense(3, activation="softmax", name="output")(x)
    # classes: 0=Normal, 1=Near-Miss, 2=Accident

    model = Model(inputs=inp, outputs=out, name="AcciDetect_MobileNet_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def preprocess_frame(frame, h, w):
    frame = cv2.resize(frame, (w, h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MobileNetV2 expects [-1, 1]
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(
        frame.astype(np.float32)
    )
    return frame


def extract_sequences_and_labels(video_path, h, w):
    print("\n  Reading video frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames in video: {total}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame, h, w))
        idx += 1
        if idx % 50 == 0:
            print(f"    Read {idx}/{total} frames...", end="\r")
    cap.release()
    print(f"  Loaded {len(frames)} frames.              ")

    if len(frames) < SEQUENCE_LENGTH + 2:
        raise ValueError("Video too short.")

    accident_start = int(len(frames) * ACCIDENT_START_RATIO)
    near_miss_start = int(len(frames) * (ACCIDENT_START_RATIO - 0.10))

    X, y = [], []
    # use stride=1 to maximise samples from short video
    for i in range(len(frames) - SEQUENCE_LENGTH):
        seq = frames[i: i + SEQUENCE_LENGTH]
        mid = i + SEQUENCE_LENGTH // 2
        if mid >= accident_start:
            label = [0, 0, 1]   # Accident
        elif mid >= near_miss_start:
            label = [0, 1, 0]   # Near-Miss
        else:
            label = [1, 0, 0]   # Normal
        X.append(seq)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    counts = y.sum(axis=0).astype(int)
    print(f"  Sequences: {len(X)}  →  Normal:{counts[0]}  Near-Miss:{counts[1]}  Accident:{counts[2]}")
    return X, y


def train(model, video_path):
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 1 — Training MobileNetV2 + LSTM")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if os.path.exists(WEIGHTS_FILE):
        print(f"  Found saved weights '{WEIGHTS_FILE}' — loading, skipping training.")
        model.load_weights(WEIGHTS_FILE)
        return model

    X, y = extract_sequences_and_labels(video_path, FRAME_HEIGHT, FRAME_WIDTH)

    # augment minority classes by repeating accident sequences
    acc_idx = np.where(y[:, 2] == 1)[0]
    nm_idx  = np.where(y[:, 1] == 1)[0]
    if len(acc_idx) > 0:
        repeat = max(1, len(X) // (len(acc_idx) * 2))
        X = np.concatenate([X] + [X[acc_idx]] * repeat + [X[nm_idx]] * max(1, repeat//2))
        y = np.concatenate([y] + [y[acc_idx]] * repeat + [y[nm_idx]] * max(1, repeat//2))
        print(f"  After augmentation: {len(X)} sequences")

    # shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = max(1, int(len(X) * 0.85))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    print(f"  Train: {len(X_tr)}   Val: {len(X_val)}")
    print(f"  Epochs: {TRAIN_EPOCHS}  (using pretrained MobileNetV2 — converges fast)\n")

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=4,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]
    )
    model.save_weights(WEIGHTS_FILE)
    print(f"\n  ✅ Weights saved → '{WEIGHTS_FILE}'")
    return model


def send_email_alert(snapshot_path, confidence, frame_number, timestamp):
    subject = f"ACCIDENT DETECTED — {CAMERA_LOCATION}"
    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px;">
      <div style="max-width:600px;margin:auto;background:#fff;border-radius:10px;
                  border-top:6px solid #e53935;padding:24px;">
        <h1 style="color:#e53935;margin-top:0;">&#128680; ACCIDENT DETECTED</h1>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr><td style="padding:8px;color:#555;font-weight:bold;">Location</td>
              <td style="padding:8px;">{CAMERA_LOCATION}</td></tr>
          <tr style="background:#fafafa;">
              <td style="padding:8px;color:#555;font-weight:bold;">Date / Time</td>
              <td style="padding:8px;">{timestamp}</td></tr>
          <tr><td style="padding:8px;color:#555;font-weight:bold;">Frame</td>
              <td style="padding:8px;">#{frame_number}</td></tr>
          <tr style="background:#fafafa;">
              <td style="padding:8px;color:#555;font-weight:bold;">CNN+LSTM Confidence</td>
              <td style="padding:8px;color:#e53935;font-weight:bold;">{confidence*100:.1f}%</td></tr>
          <tr><td style="padding:8px;color:#555;font-weight:bold;">Model</td>
              <td style="padding:8px;">MobileNetV2 (ImageNet) + LSTM (256 -&gt; 128)</td></tr>
        </table>
        <div style="margin-top:20px;padding:14px;background:#fff3f3;border-radius:6px;
                    border-left:4px solid #e53935;font-size:13px;color:#c62828;">
          Automated alert from AcciDetect AI.<br>
          Please dispatch emergency services immediately.
        </div>
      </div>
    </body></html>
    """
    msg = MIMEMultipart("related")
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
            img.add_header("Content-Disposition", "attachment",
                           filename=os.path.basename(snapshot_path))
            msg.attach(img)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        print(f"  ✅  Email sent to: {', '.join(RECIPIENT_EMAILS)}")
        return True
    except Exception as e:
        print(f"  ❌  Email failed: {e}")
        return False


def run_detection(video_path, model):
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  STEP 2 — Running Detection")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"  Video     : {video_path}  ({total} frames @ {fps:.1f} fps)")
    print(f"  Threshold : {ACCIDENT_THRESHOLD*100:.0f}%")
    print(f"  Camera    : {CAMERA_LOCATION}\n")

    buffer       = deque(maxlen=SEQUENCE_LENGTH)
    frame_number = 0
    alerts_sent  = 0
    last_alert   = 0.0
    labels       = ["Normal", "Near-Miss", "Accident"]

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_number += 1
        buffer.append(preprocess_frame(raw_frame, FRAME_HEIGHT, FRAME_WIDTH))

        if len(buffer) < SEQUENCE_LENGTH:
            continue

        seq      = np.expand_dims(np.array(buffer), axis=0)
        probs    = model.predict(seq, verbose=0)[0]
        pred     = int(np.argmax(probs))
        acc_conf = float(probs[2])

        if frame_number % 15 == 0:
            bar = "█" * int(acc_conf * 25)
            print(
                f"  Frame {frame_number:>5}/{total}  |  "
                f"{labels[pred]:<10}  |  "
                f"Accident: {acc_conf*100:5.1f}%  [{bar:<25}]"
            )

        now = time.time()
        if acc_conf >= ACCIDENT_THRESHOLD and (now - last_alert) >= COOLDOWN_SECONDS:
            last_alert   = now
            alerts_sent += 1
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n{'!'*50}")
            print(f"  ACCIDENT DETECTED  —  Frame #{frame_number}")
            print(f"  Confidence : {acc_conf*100:.1f}%")
            print(f"  Time       : {ts}")
            print(f"  Sending alert #{alerts_sent}...")
            print(f"{'!'*50}\n")

            snapshot_path = None
            if SAVE_SNAPSHOT:
                snapshot_path = f"accident_frame_{frame_number}.jpg"
                cv2.imwrite(snapshot_path, raw_frame)
                print(f"  Snapshot saved: {snapshot_path}")

            send_email_alert(snapshot_path, acc_conf, frame_number, ts)

    cap.release()
    print(f"\n{'='*50}")
    print(f"  Detection complete.")
    print(f"  Frames processed : {frame_number}")
    print(f"  Alerts sent      : {alerts_sent}")
    print(f"{'='*50}\n")


# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("\n  AcciDetect AI — MobileNetV2 + LSTM")
    print("  ====================================")
    model = build_model(SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH)
    model.summary()
    model = train(model, VIDEO_PATH)
    run_detection(VIDEO_PATH, model)