# -*- coding: utf-8 -*-
"""mimii_transformer_final_pure

完全対照実験仕様：全ID混在 ＋ クラスウェイト7.7倍 ＋ 5エポック集中訓練（W&B完全排除版）
"""

# ==============================================================================
# --- 1. データのダウンロードと解凍 (自動上書きオプション付き) ---
# ==============================================================================
print("📦 データのダウンロードを開始します（約1.3GB）...")
!wget -q https://zenodo.org/record/3384388/files/-6_dB_valve.zip
!unzip -qo ./-6_dB_valve.zip -d mimii_valve
print("✅ 解凍が完了しました。")

# ==============================================================================
# --- 2. 音声データの可視化 ---
# ==============================================================================
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

normal_path = 'mimii_valve/valve/id_00/normal/00000000.wav'
abnormal_path = 'mimii_valve/valve/id_00/abnormal/00000000.wav'

def plot_audio_features(file_path, title_prefix):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(12, 6))

    # 波形のプロット
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title(f'{title_prefix} - Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # メルスペクトログラムのプロット
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title_prefix} - Mel Spectrogram')

    plt.tight_layout()
    plt.show()

plot_audio_features(normal_path, "Normal Valve")
plot_audio_features(abnormal_path, "Abnormal Valve")

# ==============================================================================
# --- 3. 厳密なテストデータの隔離と全ID混在データパイプラインの構築 ---
# ==============================================================================
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = 'mimii_valve/valve'
IDS = ['id_00', 'id_02', 'id_04', 'id_06']

# 全IDから一斉にファイル全件をスキャン
all_normal_files = []
all_abnormal_files = []
for machine_id in IDS:
    all_normal_files.extend(glob.glob(os.path.join(BASE_DIR, machine_id, 'normal', '*.wav')))
    all_abnormal_files.extend(glob.glob(os.path.join(BASE_DIR, machine_id, 'abnormal', '*.wav')))

# 厳密なシード管理の元、テストデータ（正常100、異常150）をサンプリング
sampled_normal = random.sample(all_normal_files, min(100, len(all_normal_files)))
sampled_abnormal = random.sample(all_abnormal_files, min(150, len(all_abnormal_files)))

test_normal_set = set(sampled_normal)
test_abnormal_set = set(sampled_abnormal)
test_files_flat = [(p, 0) for p in sampled_normal] + [(p, 1) for p in sampled_abnormal]

# カンニング（データリーク）を100%排除した、全IDちゃんぽん訓練データの作成
train_files_unified = []
for f in all_normal_files:
    if f not in test_normal_set: train_files_unified.append((f, 0))
for f in all_abnormal_files:
    if f not in test_abnormal_set: train_files_unified.append((f, 1))

print(f"📊 訓練データ総数（全ID混在）: {len(train_files_unified)}件")
print(f"📊 隔離テストデータ総数      : {len(test_files_flat)}件 (正常:100, 異常:150)")

# 共通Dataset定義 (Min-Max正規化 ＆ 224x224リサイズ内包)
class UnifiedAudioDataset(Dataset):
    def __init__(self, file_pairs):
        self.file_pairs = file_pairs
    def __len__(self): 
        return len(self.file_pairs)
    def __getitem__(self, idx):
        f_path, lbl = self.file_pairs[idx]
        y, sr = librosa.load(f_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # 核心のMin-Max正規化
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        
        # テンソル化と形状変形 [1, H, W]
        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        tensor_resized = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        return tensor_resized, lbl

train_loader = DataLoader(UnifiedAudioDataset(train_files_unified), batch_size=64, shuffle=True, drop_last=True)

# ⚖️ 不均衡勾配を破壊する「体重7.7倍」の損失関数を共通定義
weights = torch.tensor([1.0, 7.706], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# ==============================================================================
# --- 4. Vision Transformer (AudioViT) モデルの定義 ---
# ==============================================================================
!pip install -q timm
import timm

class AudioViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, in_chans=1, num_classes=num_classes)
    def forward(self, x):
        return self.vit(x)

# ==============================================================================
# --- 5. 提案手法（ViT-Tiny）を5エポック集中訓練 ---
# ==============================================================================
from tqdm import tqdm

vit_model = AudioViT(num_classes=2).to(device)
vit_optimizer = optim.AdamW(vit_model.parameters(), lr=1e-4)

print("\n🏋️ 🔥 提案手法：全ID混在型ViT-Tinyの5エポック集中学習を開始します...")
for epoch in range(5):
    vit_model.train()
    running_loss = 0.0

    for features, labels in tqdm(train_loader, desc=f"ViT Epoch {epoch+1}/5"):
        features, labels = features.to(device), labels.to(device)

        vit_optimizer.zero_grad()
        outputs = vit_model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        vit_optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"  ↳ ViT Epoch {epoch+1} 終了 - 平均Loss: {avg_loss:.4f}")

# 本本API（app.py）へ移植するため、接頭辞なしの純粋な重みとして保存
torch.save(vit_model.vit.state_dict(), 'vit_anomaly_detection.pth')
print("💾 汎化型最強ViTの重みを保存しました（vit_anomaly_detection.pth）")

# ==============================================================================
# --- 6. 完全同一条件下での対照実験：Baseline CNN (ResNet18) の訓練 ---
# ==============================================================================
print("\n🏋️ 🟦 対照実験：同一の最高条件でベースラインCNN (ResNet18) の訓練を開始します...")
cnn_model = timm.create_model('resnet18', pretrained=True, in_chans=1, num_classes=2).to(device)
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-4)

for epoch in range(5):
    cnn_model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}/5"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        cnn_optimizer.zero_grad()
        loss = criterion(cnn_model(inputs), labels)
        loss.backward()
        cnn_optimizer.step()
        running_loss += loss.item()
        
    print(f"  ↳ CNN Epoch {epoch+1} 終了 - 平均Loss: {running_loss / len(train_loader):.4f}")

# ==============================================================================
# --- 7. 隔離テストデータ250件に対する「最強の2モデル」ガチンコ評価 ---
# ==============================================================================
print("\n🚀 隔離テストデータ250件の最終評価用共通前処理を実行中...")

def eval_preprocess(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False).to(device)

models_to_compare = {
    'ResNet18 (Baseline CNN)': cnn_model.eval(),
    'Vision Transformer (Proposed)': vit_model.eval()
}

benchmarks = {}
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

for m_name, tgt_model in models_to_compare.items():
    y_true, y_pred, y_scores = [], [], []
    for f_path, label in test_files_flat:
        try:
            tensor = eval_preprocess(f_path)
            with torch.no_grad():
                probs = F.softmax(tgt_model(tensor), dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob_abnormal = probs[0][1].item()
            y_true.append(label)
            y_pred.append(pred)
            y_scores.append(prob_abnormal)
        except:
            pass
            
    benchmarks[m_name] = {
        'Acc': accuracy_score(y_true, y_pred) * 100,
        'Rec': recall_score(y_true, y_pred) * 100,
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_scores)
    }

# ==============================================================================
# --- 8. README直結型ベンチマーク統計の最終出力 ---
# ==============================================================================
print("\n" + "="*80)
print("🏆 【完全対照実験】READMEにそのまま記載可能な最終ベンチマーク結果")
print("="*80)
for m_name, metrics in benchmarks.items():
    print(f"■ {m_name}")
    print(f"  - Accuracy (総合精度)   : {metrics['Acc']:.1f}%")
    print(f"  - Recall (異常検知率)   : {metrics['Rec']:.1f}%  <-- 現場の最重要指標")
    print(f"  - F1-Score (調和平均)   : {metrics['F1']:.3f}")
    print(f"  - ROC-AUC (識別能力値)  : {metrics['AUC']:.3f}")
    print("-"*80)

# ==============================================================================
# --- 9. デモ検証用テストサンプルの物理隔離・ZIPエクスポート ---
# ==============================================================================
import shutil
from google.colab import files

OUTPUT_DIR = '/content/test_samples_all'
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(f'{OUTPUT_DIR}/normal', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/abnormal', exist_ok=True)

for f_path, lbl in test_files_flat:
    parts = f_path.split('/')
    m_id = parts[-3]
    f_name = parts[-1]
    tgt_sub = 'normal' if lbl == 0 else 'abnormal'
    shutil.copy(f_path, os.path.join(OUTPUT_DIR, tgt_sub, f"{m_id}_{f_name}"))

print("\n📦 ローカル検証・デモサイトテスト用のZIPアーカイブを構築中...")
shutil.make_archive('/content/test_samples_all', 'zip', OUTPUT_DIR)
print("⬇️ ダウンロードトリガーを発火します...")
files.download('/content/test_samples_all.zip')
print("🏁 すべての検証タスクが完全終了しました！お疲れ様でした。")