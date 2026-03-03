import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 設定網頁標題
st.set_page_config(page_title="酒類分類預測系統", layout="wide")

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# 2. 側邊欄設計
st.sidebar.header("模型選擇與資訊")

# 模型選擇下拉選單
model_option = st.sidebar.selectbox(
    "請選擇預測模型：",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

# 顯示資料集資訊
st.sidebar.markdown("---")
st.sidebar.subheader("🍷 酒類資料集資訊")
st.sidebar.info(f"""
- **特徵數量**: {len(wine_data.feature_names)}
- **樣本總數**: {len(df)}
- **類別數量**: {len(wine_data.target_names)} ({', '.join(wine_data.target_names)})
""")

# 3. 主要內容區域
st.title("🍷 酒類產品分類預測系統")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 資料集前 5 筆內容")
    st.write(df.head())

with col2:
    st.subheader("📊 特徵統計資訊")
    st.write(df.describe())

st.markdown("---")

# 4. 預測邏輯
st.subheader(f"🚀 使用 {model_option} 進行預測")

if st.button("執行預測"):
    # 準備資料
    X = wine_data.data
    y = wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 選擇模型
    if model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_option == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=3000)
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_option == "隨機森林":
        model = RandomForestClassifier(n_estimators=100)

    # 訓練模型
    with st.spinner(f"正在訓練 {model_option} 模型..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    # 顯示結果
    st.success(f"預測完成！")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="模型準確度 (Accuracy)", value=f"{accuracy:.2%}")
    
    with res_col2:
        st.write("測試集預測前 10 筆結果：")
        st.write(pd.DataFrame({"實際值": y_test[:10], "預測值": y_pred[:10]}))
        
    st.balloons()
