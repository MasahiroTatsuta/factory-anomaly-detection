import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# --- 1. モデルの定義 ---
class AudioViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, in_chans=1, num_classes=num_classes)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit(x)

# グローバル変数としてモデルとデバイスを準備
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 起動時（Lifespan）の処理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = AudioViT(num_classes=2)
    
    # モデルの重みを読み込む（Colabからダウンロードしたファイルと同じ階層に置いている想定）
    model_path = 'vit_anomaly_detection.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️ Warning: '{model_path}' not found. API will run without a trained model.")
    
    yield # ここでAPIが稼働します
    
    print("🛑 API shut down.")

# FastAPIアプリの初期化（lifespanを登録）
app = FastAPI(title="Factory Anomaly Detection API", lifespan=lifespan)

# --- 3. 前処理関数の定義 ---
# def preprocess_audio(file_bytes):
#     y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000)
#     mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
#     mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
#     feature_tensor = torch.tensor(mel_spect_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     return feature_tensor

# FastAPIのコード（app.pyなど）の前処理部分の修正イメージ
def preprocess_audio(file_bytes):
    # 1. 音声のロード
    y, sr = librosa.load(file_bytes, sr=16000)
    
    # 2. メルスペクトログラム変換
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # 🔥【★超重要★】Colabと完全に同じMin-Max正規化をここに挿入！
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    
    # 3. テンソル化とリサイズ
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor_resized = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
    return tensor_resized

# --- 4. 予測エンドポイント ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    
    try:
        file_bytes = await file.read()
        features = preprocess_audio(file_bytes)
        features = features.to(device)
        
        with torch.no_grad():
            outputs = model(features)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        normal_prob = float(probabilities[0])
        abnormal_prob = float(probabilities[1])
        prediction = "Normal" if normal_prob > abnormal_prob else "Abnormal"
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": prediction,
            "confidence": {
                "normal": round(normal_prob * 100, 2),
                "abnormal": round(abnormal_prob * 100, 2)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "healthy", "model": "ViT-Tiny"}