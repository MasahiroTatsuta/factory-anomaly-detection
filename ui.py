import streamlit as st
import requests

# ページの基本設定
st.set_page_config(page_title="Factory Anomaly Detection", page_icon="🏭")

st.title("🏭 Factory Anomaly Detection")
st.write("工場の機械音（.wav）をアップロードすると、AIが正常か異常かを判定します。")

# ファイルアップローダー
uploaded_file = st.file_uploader("音声ファイルを選択してください", type=["wav"])

if uploaded_file is not None:
    # アップロードされた音声を画面上で再生できるようにする
    st.audio(uploaded_file, format="audio/wav")
    
    # 判定ボタン
    if st.button("判定を実行", type="primary"):
        with st.spinner("AIが推論中..."):
            # バックエンド（FastAPI）のURL
            url = "http://127.0.0.1:8000/predict"
            
            # APIへ送信するデータを作成
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
            
            try:
                # APIへPOSTリクエストを送信
                response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    conf_normal = result["confidence"]["normal"]
                    conf_abnormal = result["confidence"]["abnormal"]
                    
                    st.divider()
                    st.subheader("📊 判定結果")
                    
                    # 結果に応じてUIの表示を変える
                    if prediction == "Normal":
                        st.success(f"✅ 判定: 正常 (Normal)")
                    else:
                        st.error(f"⚠️ 判定: 異常 (Abnormal)")
                        
                    # 確信度をプログレスバーで視覚的に表示
                    st.write(f"**正常スコア: {conf_normal}%**")
                    st.progress(conf_normal / 100)
                    
                    st.write(f"**異常スコア: {conf_abnormal}%**")
                    st.progress(conf_abnormal / 100)
                    
                else:
                    st.error(f"APIエラー: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("APIサーバーに接続できません。FastAPI（uvicorn）が裏で起動しているか確認してください。")