#!/usr/bin/env python
"""
JARVIS 啟動腳本
啟動 Django 後端和 Chainlit 前端
"""
import os
import sys
import subprocess
import time
import webbrowser
from threading import Thread

# 網址 & 端口設置
DJANGO_HOST = "localhost"
DJANGO_PORT = "8000"
CHAINLIT_PORT = "8501"

def run_django_server():
    """啟動 Django 服務器"""
    print("啟動 Django 後端服務...")
    # 設置環境變量
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "jarvis_project.settings")
     
    # 使用子進程運行
    subprocess.run([sys.executable, "manage.py", "runserver", f"{DJANGO_HOST}:{DJANGO_PORT}"])


def run_chainlit_app():
    """啟動 Chainlit 應用"""
    print("啟動 Chainlit 前端界面...")
    os.chdir("chainlit_app")
    # 使用標準應用 (已內置語音功能)
    subprocess.run([sys.executable, "-m", "chainlit", "run", "app.py", "--port", CHAINLIT_PORT])


def main():
    """主函數"""
    # 設定 Django API 基礎 URL 環境變量
    os.environ["DJANGO_API_BASE_URL"] = f"http://{DJANGO_HOST}:{DJANGO_PORT}/api/v1"
    
    # 檢查是否已安裝所需的軟件包
    try:
        import django
        import chainlit
        import openai
        import requests
        import sentence_transformers
        import yfinance  # 確保安裝了 yfinance 包
        import llama_index  # 確保安裝了 llama_index 包
    except ImportError as e:
        print(f"缺少必要的依賴包: {e}")
        print("請先運行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 檢查.env文件
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("未找到.env文件。請從.env.example創建一個.env文件並填入您的API密鑰。")
        else:
            print("未找到.env或.env.example文件。請創建.env文件並設置必要的環境變量。")
    
    # 確保必要的目錄存在
    os.makedirs("data/knowledge_base", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)  # 確保快取目錄存在
    
    # 創建並啟動Django服務器線程
    django_thread = Thread(target=run_django_server)
    django_thread.daemon = True
    django_thread.start()
    
    # 等待Django服務器啟動
    print("等待Django服務器啟動...")
    connection_successful = False
    max_attempts = 20
    attempt = 0
    
    while attempt < max_attempts and not connection_successful:
        attempt += 1
        try:
            import requests
            response = requests.get(f"http://{DJANGO_HOST}:{DJANGO_PORT}/api/v1/health/", timeout=3)
            if response.status_code == 200:
                print(f"Django後端已成功啟動!")
                connection_successful = True
            else:
                print(f"Django後端回應狀態碼: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"等待Django服務器...嘗試 {attempt}/{max_attempts}")
            time.sleep(2)
    
    if not connection_successful:
        print("無法連接到Django後端，繼續啟動Chainlit前端，但功能可能受限...")
    
    # 啟動Chainlit應用
    print("\n==============================")
    print("現在啟動Chainlit前端界面...")
    print("==============================\n")
    run_chainlit_app()

if __name__ == "__main__":
    print("\n==============================")
    print("歡迎使用JARVIS AI助理!")
    print("支援RAG功能，可輸入 /rag on 啟用知識庫參考")
    print("支援網路搜索功能，可輸入 /web on 啟用")
    print("支援回應快取功能，可輸入 /cache on 啟用(預設開啟)")
    print("支援快取統計分析，可輸入 /cache stats 查看")
    print("支援內容安全過濾，可輸入 /safety on 啟用(預設開啟)")
    print("支援內容過濾級別設置，可輸入 /safety level low/medium/high")
    print("支援語音輸入輸出，可使用麥克風按鈕進行語音對話")
    print("支援股票分析功能，可輸入 /stock 股票代碼 查詢股票資訊")
    print("==============================\n")
    main()
