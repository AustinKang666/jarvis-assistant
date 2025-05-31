"""
JARVIS助理 - 支援語音輸入及股票查詢功能
"""
import os
import sys
import requests
import json
import shutil
import io
import traceback
import time
import re
from typing import List, Dict, Any

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import chainlit as cl

# Django API 基礎URL - 從環境變量獲取或使用默認值
DJANGO_API_BASE_URL = os.environ.get("DJANGO_API_BASE_URL", "http://127.0.0.1:8000/api/v1")
print(f"使用 Django API 基礎 URL: {DJANGO_API_BASE_URL}")

@cl.on_chat_start
async def on_chat_start():
    """當用戶開始新對話時執行"""
    # 創建會話ID
    import uuid
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)
    
    # 初始化消息歷史
    cl.user_session.set("message_history", [])
    
    # 初始化RAG開關（預設啟用）
    cl.user_session.set("use_rag", True)
    
    # 初始化網路搜索開關（預設關閉）
    cl.user_session.set("use_web_search", False)
    
    # 初始化快取開關（預設啟用）
    cl.user_session.set("use_cache", True)
    
    # 初始化安全過濾開關（預設啟用）
    cl.user_session.set("use_safety", True)
    
    # 初始化安全過濾級別（預設中級）
    cl.user_session.set("safety_level", "medium")
    
    # 進行環境檢查並使用重試機制
    api_available = False
    max_retries = 6  # 增加重試次數
    retry_delay = 5  # 等待更長時間
    
    for attempt in range(max_retries):
        try:
            # 檢查 Django API 是否可用
            print(f"正在檢查 Django API 連接... {DJANGO_API_BASE_URL}/health/ (嘗試 {attempt+1}/{max_retries})")
            health_check = requests.get(f"{DJANGO_API_BASE_URL}/health/", timeout=10)  # 給它更多時間響應
            if health_check.status_code == 200:
                api_available = True
                print(f"Django API 健康檢查成功")
                break
            else:
                print(f"Django API 回應非200狀態碼: {health_check.status_code}")
        except Exception as e:
            print(f"Django API 連接失敗: {e}")
        
        if attempt < max_retries - 1:
            print(f"將在 {retry_delay} 秒後重試...")
            time.sleep(retry_delay)
    
    # 為歡迎訊息準備元素
    welcome_elements = [
        cl.Text(name="voice_status", content="提示：使用麥克風按鈕進行語音輸入")
    ]
    
    # 為提示消息添加RAG說明
    welcome_elements.append(
        cl.Text(name="rag_status", content="提示：使用 `/rag on` 或 `/rag off` 來開啟或關閉知識庫參考功能")
    )
    
    # 為提示消息添加網路搜索說明
    welcome_elements.append(
        cl.Text(name="web_search_status", content="提示：使用 `/web on` 或 `/web off` 來開啟或關閉網路搜索功能")
    )
    
    # 為提示消息添加快取說明
    welcome_elements.append(
        cl.Text(name="cache_status", content="提示：使用 `/cache on` 或 `/cache off` 來開啟或關閉回答快取功能")
    )
    
    # 為提示消息添加安全過濾說明
    welcome_elements.append(
        cl.Text(name="safety_status", content="提示：使用 `/safety on` 或 `/safety off` 來開啟或關閉安全過濾功能")
    )
    
    # 為提示消息添加股票查詢說明
    welcome_elements.append(
        cl.Text(name="stock_status", content="提示：使用 `/stock 股票代碼` 來查詢股票資訊，例如 `/stock 2330` 或 `/stock AAPL`")
    )
    
    # 準備歡迎訊息
    welcome_msg = (
        "您好！我是 JARVIS，您的個人 AI 助理。\n\n"
        "● 您可以直接輸入問題\n"
        "● 使用麥克風按鈕進行語音輸入\n"
        "● 在左下角使用回形針按鈕來上傳文件\n"
        "● 您上傳的文件會自動加入知識庫供我參考\n"
        "● 使用 `/web on` 可以啟用網路搜索功能\n"
        "● 系統預設啟用回答快取，可加快回答相似問題的速度\n"
        "● 安全過濾功能已啟用，可防止不當內容輸出\n"
        "● 使用 `/stock 股票代碼` 進行股票分析，例如 `/stock 2330`"
    )
    
    # 檢查 Django API 是否可用
    if not api_available:
        welcome_msg += "\n\n 無法連接到後端服務，部分功能可能無法使用。"
        welcome_elements.append(cl.Text(name="api_warning", content="警告：後端服務未啟動，請在命令行執行 python manage.py runserver 8000"))
    
    # 發送歡迎訊息
    await cl.Message(
        content=welcome_msg,
        elements=welcome_elements
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """處理用戶發送的訊息"""
    # 獲取會話數據
    conversation_id = cl.user_session.get("conversation_id")
    message_history = cl.user_session.get("message_history")
    use_rag = cl.user_session.get("use_rag")
    use_web_search = cl.user_session.get("use_web_search")
    use_cache = cl.user_session.get("use_cache")
    use_safety = cl.user_session.get("use_safety")
    safety_level = cl.user_session.get("safety_level")
    
    # 檢查是否是檔案上傳消息 - 檢查 message.elements 中的附件
    if message.elements:
        # 列印詳細信息以便診斷
        print(f"收到含有元素的消息，數量: {len(message.elements)}")
        
        # 遍歷所有元素處理文件(使用者上傳於Chainlit UI)
        for elem in message.elements:
            print(f"處理元素類型: {type(elem)}")
            await process_file(elem)
        return
    

    # ⭐檢查是否是股票查詢指令 [此單元新增的地方]
    import re # 匯入正則表達式模組，用來進行字串格式比對與抽取
    # 📌 定義指令的正則表達式樣式
    # 📥 使用者輸入若為 "/stock XXX"，這段會解析出後面的股票代碼（支援英文+數字）
    # 說明：
    #   ^         → 開頭必須是
    #   /stock    → 字面字串 "/stock"
    #   \s+       → 至少一個空白（允許 "/stock 2330" 或 "/stock   TSLA"）
    #   ([A-Za-z0-9]+) → 將空白後的字元群（股票代碼）捕捉起來（含大小寫英文與數字）
    #   $         → 字串結尾
    stock_pattern = r"^/stock\s+([A-Za-z0-9]+)$"
    # 🔍 嘗試將使用者輸入內容與 stock_pattern 做比對
    # 若符合（例如輸入 "/stock AAPL"），就會回傳 match 物件；否則回傳 None
    stock_match = re.match(stock_pattern, message.content)
    
    
    # [ .group(0)、.group(1) 說明 ]:
    # ^/stock\s+([A-Za-z0-9]+)$
    #                 👆    👈 這個括號就是 group(1)
    #  match.group(0) ➜ /stock AAPL（整段）
    #  match.group(1) ➜ AAPL（我們要的股票代碼！）

    # 如果成功比對（代表使用者輸入的是 "/stock XXX" 的格式）:
    if stock_match:
        stock_symbol = stock_match.group(1) # 透過 group(1) 抓取第一個括號群組對應到的股票代碼（AAPL、2330 等）
        await analyze_stock(stock_symbol)  # 呼叫定義的分析函式（將解析出來的股票代碼傳入） => 就是此程式後面的 async def analyze_stock()
        return  # 呼叫完分析後直接 return（代表這次訊息已處理完畢，不會往下走其他 handler）
    


    # 處理指令 - 切換RAG模式
    if message.content.lower() in ["/rag on", "rag on"]:
        cl.user_session.set("use_rag", True)
        await cl.Message(content="RAG功能已啟用，我將參考您上傳的文件來回答問題。").send()
        return
    elif message.content.lower() in ["/rag off", "rag off"]:
        cl.user_session.set("use_rag", False)
        await cl.Message(content="RAG功能已停用，我將不會參考您上傳的文件來回答問題。").send()
        return
    # 處理指令 - 切換網路搜索模式
    elif message.content.lower() in ["/web on", "web on"]:
        cl.user_session.set("use_web_search", True)
        await cl.Message(content="網路搜索功能已啟用，我將在知識庫不足時使用網路搜索。").send()
        return
    elif message.content.lower() in ["/web off", "web off"]:
        cl.user_session.set("use_web_search", False)
        await cl.Message(content="網路搜索功能已停用，我將只使用內部知識。").send()
        return
    # 處理指令 - 切換快取模式
    elif message.content.lower() in ["/cache on", "cache on"]:
        cl.user_session.set("use_cache", True)
        await cl.Message(content="回答快取功能已啟用，我將快速回答相似的問題。").send()
        return
    elif message.content.lower() in ["/cache off", "cache off"]:
        cl.user_session.set("use_cache", False)
        await cl.Message(content="回答快取功能已停用，我將重新處理每個問題。").send()
        return
    # 處理指令 - 快取統計
    elif message.content.lower() in ["/cache stats", "cache stats"]:
        try:
            # 呼叫快取統計API
            stats_url = f"{DJANGO_API_BASE_URL}/cache_stats/"
            response = requests.get(stats_url)  # 移除超時限制
            if response.status_code == 200:
                stats = response.json().get("stats", {})
                stats_text = f"快取統計信息:\n"
                stats_text += f"- 快取總條目數: {stats.get('total_entries', 0)}\n"
                stats_text += f"- 快取檔案大小: {stats.get('cache_size_bytes', 0) // 1024} KB\n"
                
                # 顯示各類型的快取數量
                if 'source_type_counts' in stats:
                    stats_text += "\n快取來源分佈:\n"
                    for source, count in stats.get('source_type_counts', {}).items():
                        source_name = {
                            "direct": "直接回答",
                            "rag": "知識庫增強",
                            "web_search": "網路搜索"
                        }.get(source, source)
                        stats_text += f"- {source_name}: {count}\n"
                
                # 顯示最多訪問的問題
                if 'most_accessed' in stats and stats['most_accessed']:
                    most_accessed = stats['most_accessed']
                    stats_text += f"\n最多訪問的問題: {most_accessed.get('question', '')}\n"
                    stats_text += f"- 訪問次數: {most_accessed.get('access_count', 0)}\n"
                
                await cl.Message(content=stats_text).send()
            else:
                await cl.Message(content=f"獲取快取統計失敗: {response.status_code}").send()
        except Exception as e:
            await cl.Message(content=f"獲取快取統計時出錯: {str(e)}").send()
        return
    # 處理指令 - 清除快取
    elif message.content.lower() in ["/cache clear", "cache clear"]:
        try:
            # 呼叫清除快取API
            clear_url = f"{DJANGO_API_BASE_URL}/clear_cache/"
            response = requests.post(clear_url)  # 移除超時限制
            if response.status_code == 200:
                await cl.Message(content="快取已成功清除。").send()
            else:
                await cl.Message(content=f"清除快取失敗: {response.status_code}").send()
        except Exception as e:
            await cl.Message(content=f"清除快取時出錯: {str(e)}").send()
        return
    # 處理指令 - 切換安全過濾模式
    elif message.content.lower() in ["/safety on", "safety on"]:
        cl.user_session.set("use_safety", True)
        await cl.Message(content="安全過濾功能已啟用，我將確保所有回應皆符合安全標準。").send()
        return
    elif message.content.lower() in ["/safety off", "safety off"]:
        cl.user_session.set("use_safety", False)
        await cl.Message(content="安全過濾功能已停用，請責任使用。").send()
        return
    # 處理指令 - 設置安全過濾級別
    elif message.content.lower() in ["/safety low", "safety low"]:
        cl.user_session.set("safety_level", "low")
        await cl.Message(content="安全過濾級別已設置為「低」，僅過濾明確有害內容。").send()
        return
    elif message.content.lower() in ["/safety medium", "safety medium"]:
        cl.user_session.set("safety_level", "medium")
        await cl.Message(content="安全過濾級別已設置為「中」，將過濾有害內容及可能有爭論的內容。").send()
        return
    elif message.content.lower() in ["/safety high", "safety high"]:
        cl.user_session.set("safety_level", "high")
        await cl.Message(content="安全過濾級別已設置為「高」，將使用最嚴格的安全標準。").send()
        return
    # 處理指令 - 安全過濾器狀態
    elif message.content.lower() in ["/safety status", "safety status"]:
        try:
            # 呼叫安全過濾器狀態 API
            status_url = f"{DJANGO_API_BASE_URL}/safety_config/"
            response = requests.get(status_url)  # 移除超時限制
            if response.status_code == 200:
                safety_config = response.json().get("safety_config", {})
                status_text = f"安全過濾器狀態:\n"
                status_text += f"- 狀態: {'啟用' if safety_config.get('enabled', True) else '停用'}\n"
                status_text += f"- 過濾級別: {safety_config.get('filter_level', 'medium')}\n"
                status_text += f"- 使用模型: {safety_config.get('model', 'gpt-4o-mini')}"
                
                await cl.Message(content=status_text).send()
            else:
                await cl.Message(content=f"獲取安全過濾器狀態失敗: {response.status_code}").send()
        except Exception as e:
            await cl.Message(content=f"獲取安全過濾器狀態時出錯: {str(e)}").send()
        return
    # 處理指令 - 語音功能說明
    elif message.content.lower() in ["/voice", "voice"]:
        await cl.Message(content="語音功能使用說明:\n\n1. 點擊輸入框左側的麥克風圖標開始錄音\n2. 對著麥克風說話\n3. 點擊停止按鈕或等待自動停止\n4. 系統會自動識別您的語音並將其轉換為文字").send()
        return
    # 處理指令 - 股票功能說明
    elif message.content.lower() in ["/stock", "stock"]:
        await cl.Message(content="股票分析功能使用說明:\n\n1. 輸入 `/stock 股票代碼` 來查詢股票資訊\n2. 台股請直接輸入數字代碼，例如：`/stock 2330`\n3. 美股請輸入股票代碼，例如：`/stock AAPL`\n4. 系統將提供股票的完整分析，包括價格、基本面、財務比率與買賣建議").send()
        return
    
    # 顯示思考中的狀態
    thinking_msg = cl.Message(content="思考中...")
    await thinking_msg.send()
    
    # 準備請求數據
    request_data = {
        "message": message.content,
        "conversation_id": conversation_id,
        "message_history": message_history,
        "use_rag": use_rag,
        "use_web_search": use_web_search,
        "use_cache": use_cache,
        "use_safety": use_safety,
        "safety_level": safety_level
    }
    
    try:
        # 調用Django API
        api_url = f"{DJANGO_API_BASE_URL}/jarvis/"
        print(f"正在調用 API: {api_url}")
        response = requests.post(
            api_url,
            json=request_data,
            headers={"Content-Type": "application/json"}
            # 移除超時限制，允許處理更複雜的請求
        )
        
        # 移除思考中的消息
        await thinking_msg.remove()
        
        # 檢查響應
        if response.status_code == 200:
            response_data = response.json()
            
            # 獲取回應文本
            response_text = response_data.get("message", "")
            
            # 獲取更新後的消息歷史
            message_history = response_data.get("message_history", [])
            cl.user_session.set("message_history", message_history)
            
            # 發送回應
            elements = []
            
            # 如果使用了RAG，顯示相關提示
            if use_rag and "(回答參考了您上傳的知識庫資料)" in response_text:
                response_text = response_text.replace("(回答參考了您上傳的知識庫資料)", "")
                elements.append(cl.Text(name="rag_used", content="✓ 回答使用了您上傳的知識庫資料"))
            
            # 如果使用了網路搜索，顯示相關提示
            if use_web_search and "(回答參考了知識庫資料及網路搜索結果)" in response_text:
                response_text = response_text.replace("(回答參考了知識庫資料及網路搜索結果)", "")
                elements.append(cl.Text(name="web_search_used", content="✓ 回答使用了網路搜索結果"))
            
            # 如果回答來自快取，顯示相關提示
            if "(回答來自快取)" in response_text:
                response_text = response_text.replace("(回答來自快取)", "")
                elements.append(cl.Text(name="cache_used", content="✓ 回答來自快取"))
            
            # 如果回答來自快取(相似問題)，顯示相關提示
            if "(回答來自快取 - 基於相似問題" in response_text:
                # 尋找和提取相似度值
                match = re.search(r"回答來自快取 - 基於相似問題，相似度: (\d+\.\d+)", response_text)
                if match:
                    similarity = match.group(1)
                    response_text = re.sub(r"\(回答來自快取 - 基於相似問題，相似度: \d+\.\d+\)", "", response_text)
                    elements.append(cl.Text(name="similar_question_used", content=f"✓ 回答使用了相似問題的快取 (相似度: {similarity})"))
            
            # 如果答案經過安全過濾，顯示相關提示
            if "(該回應已經過安全審核調整)" in response_text:
                response_text = response_text.replace("(該回應已經過安全審核調整)", "")
                elements.append(cl.Text(name="safety_filtered", content="✓ 回答經過安全過濾器調整"))
            
            # 發送文本回應
            text_message = await cl.Message(content=response_text, elements=elements).send()
            
            # 語音輸出功能已移除 - 只保留語音輸入功能
            pass
            
        else:
            # 處理錯誤
            error_message = f"API請求失敗: {response.status_code}"
            try:
                error_data = response.json()
                error_message += f" - {error_data.get('message', '')}"
            except:
                error_message += f" - {response.text}"  # 拿到伺服器原始文字
                               
            await cl.Message(content=error_message).send()
            
    except requests.exceptions.Timeout:
        await cl.Message(content="API請求超時，請稍後再試或檢查伺服器狀態。").send()
    except requests.exceptions.ConnectionError:
        await cl.Message(content="無法連接到後端服務，請確保伺服器正在運行。").send()
    except Exception as e:
        error_message = f"發生錯誤: {str(e)}"
        print(f"詳細錯誤: {traceback.format_exc()}")
        await cl.Message(content=error_message).send()



async def analyze_stock(stock_symbol: str):
    """處理股票分析請求（Chainlit用戶輸入後觸發）"""

    # 建立提示訊息，告知使用者分析中（Chainlit UI 上會顯示這段文字）
    thinking_msg = cl.Message(content=f"正在分析 {stock_symbol} 股票，請稍候...")
    await thinking_msg.send()
    
    try:
        # 設定分析用的 Django API 路徑（對應 views.py 中的 analyze_stock 端點）=> views.py 內的 def analyze_stock()
        api_url = f"{DJANGO_API_BASE_URL}/analyze_stock/"
        response = requests.post(
            api_url,
            json={"stock_symbol": stock_symbol},  # 傳送使用者輸入的股票代碼 [已轉換成指定型式 => 透過 async def on_message() 的正則表示式]
        )
        
        # 移除思考中的消息
        await thinking_msg.remove()
        
        # 如果 HTTP 回傳狀態為 200，表示分析成功 => 分析成功時才會回傳狀態碼為 200  => 可看 views.py 內的 def analyze_stock()
        if response.status_code == 200:
            result = response.json()  # 解析 JSON 回傳內容
            analysis = result.get("analysis", "無法取得分析結果")  # 取得分析結果內容
            # 將分析結果顯示於 UI 上（以 Message 形式回覆使用者）       
            await cl.Message(content=f"{stock_symbol} 分析結果：\n\n{analysis}").send()

        #  若 HTTP 狀態碼非 200，代表請求失敗（如 stock_symbol 無效）=> 可看 views.py 內的 def analyze_stock() 的 return
        else:
            try:
                error_data = response.json()  # 解析 JSON 回傳內容
                error_msg = error_data.get("message", "股票分析請求失敗")  # 嘗試讀取錯誤訊息
            
            # 若後端回傳非 JSON（可能是 500 或 HTML 錯誤頁），則 fallback
            except:
                error_msg = f"股票分析請求失敗：狀態碼 {response.status_code}"
            
            # 將錯誤訊息顯示在使用者介面
            await cl.Message(content=error_msg).send()

    # 若發生非預期錯誤（如 requests.post 執行中斷、主機無回應等），顯示錯誤訊息
    except Exception as e:
        await thinking_msg.remove()  # 預防異常情況下 loading 訊息未移除
        await cl.Message(content=f"股票分析時發生錯誤: {str(e)}").send()  # 將錯誤訊息顯示在使用者介面
    
    finally:
        #  確保最後一定會移除「分析中...」訊息，無論 try/except 是否發生
        #   → 即使重複移除也不會拋錯，Chainlit 會自動忽略不存在的訊息 (保險機制)
        await thinking_msg.remove()




async def process_file(file):
    """處理上傳的文件"""
    temp_file_path = None
    
    try:
        # 檢查是否是文件類型
        if not hasattr(file, 'name'):
            print(f"跳過非文件元素: {type(file)}")
            return
            
        # 輸出檔案資訊以檢查
        print(f"處理檔案: {file.name}")
        print(f"檔案屬性: {dir(file)}")
        for attr in ['path', 'content', 'get_bytes', 'read']:
            print(f"  Has {attr}? {hasattr(file, attr)}")
        
        # 顯示正在處理文件的消息
        processing_msg = cl.Message(content=f"正在處理您上傳的文件: {file.name}...")
        await processing_msg.send()
        
        # 檢查文件類型
        file_ext = os.path.splitext(file.name)[1].lower()
        
        # 檢查是否為圖像文件
        image_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        is_image = file_ext in image_formats
        
        if not is_image and file_ext not in ['.pdf', '.txt', '.docx', '.doc']:
            await cl.Message(content=f"不支持的文件類型: {file_ext}，請上傳 PDF、TXT、Word 文檔或圖像文件").send()
            return
        
        # 創建臨時目錄（如果不存在）
        temp_dir = os.path.join(project_root, "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成安全的文件名
        import re
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', file.name)
        temp_file_path = os.path.join(temp_dir, safe_filename)
        
        # 保存上傳的文件，嘗試多種方法
        file_saved = False
        file_data = None
        
        # 方法 1: 嘗試使用 get_bytes（新版 Chainlit）
        if hasattr(file, 'get_bytes'):
            try:
                file_data = await file.get_bytes()
                with open(temp_file_path, "wb") as f:
                    f.write(file_data)
                file_saved = True
                print("使用 get_bytes 方法保存文件成功")
            except Exception as e:
                print(f"使用 get_bytes 方法失敗: {e}")
                
        # 方法 2: 嘗試從 content 屬性獲取
        if not file_saved and hasattr(file, 'content') and file.content:
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(file.content)
                file_saved = True
                print("使用 content 屬性保存文件成功")
            except Exception as e:
                print(f"使用 content 屬性失敗: {e}")
                
        # 方法 3: 嘗試從 path 屬性讀取
        if not file_saved and hasattr(file, 'path') and os.path.exists(file.path):
            try:
                shutil.copy(file.path, temp_file_path)
                file_saved = True
                print("使用 path 屬性保存文件成功")
            except Exception as e:
                print(f"使用 path 屬性失敗: {e}")
                
        # 方法 4: 嘗試使用 read 方法
        if not file_saved and hasattr(file, 'read'):
            try:
                content = file.read()
                with open(temp_file_path, "wb") as f:
                    if isinstance(content, str):
                        f.write(content.encode('utf-8'))
                    else:
                        f.write(content)
                file_saved = True
                print("使用 read 方法保存文件成功")
            except Exception as e:
                print(f"使用 read 方法失敗: {e}")
                
        # 如果所有方法都失敗，顯示診斷信息
        if not file_saved:
            file_attrs = dir(file)
            print(f"所有文件屬性: {file_attrs}")
            await cl.Message(content=f"無法讀取文件內容，請嘗試其他文件或聯繫管理員。").send()
            return
                
        # 將文件上傳到Django API
        if os.path.exists(temp_file_path):
            upload_success = False
            try:
                with open(temp_file_path, "rb") as f:
                    files = {"file": (safe_filename, f)}
                    
                    # 依據文件類型選擇API端點
                    if is_image:
                        upload_url = f"{DJANGO_API_BASE_URL}/analyze_image/"
                        print(f"正在上傳圖像 {safe_filename} 到分析API: {upload_url}")
                    else:
                        upload_url = f"{DJANGO_API_BASE_URL}/upload/"
                        print(f"正在上傳文件 {safe_filename} 到Django API: {upload_url}")
                    
                    # 使用較長的超時時間處理大文件
                    response = requests.post(
                        upload_url,
                        files=files
                        # 移除超時限制，允許處理大型文件上傳
                    )
                    
                    # 檢查響應
                    if response.status_code == 200:
                        upload_success = True
                        response_data = response.json()
                        success_message = response_data.get("message", "文件已成功處理")
                        
                        # 顯示成功消息
                        await cl.Message(content=success_message).send()
                        
                        # 如果是圖像分析，顯示分析結果
                        if is_image and "analysis" in response_data:
                            analysis_result = response_data.get("analysis", "")
                            # 顯示分析結果
                            await cl.Message(content=f"圖像分析結果:\n\n{analysis_result}").send()
                        
                        # 確保RAG模式已開啟
                        cl.user_session.set("use_rag", True)
                        
                        # 添加提示消息
                        await cl.Message(
                            content="我現在可以根據這個文件回答問題了。請隨時提問！",
                            elements=[
                                cl.Text(name="rag_reminder", content="提示：您可以通過輸入 `/rag off` 或 `/rag on` 來關閉或開啟知識庫參考功能。")
                            ]
                        ).send()

                    else:
                        error_message = f"文件上傳失敗: {response.status_code}"
                        try:
                            error_data = response.json()
                            error_message += f" - {error_data.get('message', '')}"
                        except:
                            error_message += f" - {response.text}"
                        
                        await cl.Message(content=error_message).send()
                        
            except requests.exceptions.Timeout:  # 特定錯誤類型：上傳逾時
                await cl.Message(content="文件上傳超時，可能是文件太大或伺服器處理能力有限。").send()
            except requests.exceptions.ConnectionError:  # 特定錯誤類型：連不到主機
                await cl.Message(content="無法連接到後端服務，請確保伺服器正在運行。").send()
            except Exception as e: # 所有其他錯誤
                error_message = f"上傳文件時發生錯誤: {str(e)}"
                print(f"詳細錯誤: {traceback.format_exc()}")
                await cl.Message(content=error_message).send()
            finally:
                # 無論成功與否，都清理臨時文件
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        print(f"已刪除臨時文件: {temp_file_path}")
                except Exception as e:
                    print(f"刪除臨時文件時出錯: {e}")
        else:
            await cl.Message(content=f"雖然文件內容已讀取，但無法保存到臨時目錄。").send()
    
    except Exception as e:
        error_message = f"處理文件時發生錯誤: {str(e)}"
        print(f"詳細錯誤: {traceback.format_exc()}")
        await cl.Message(content=error_message).send()
        
        # 確保清理臨時文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


if __name__ == "__main__":
    import os
    os.environ["DJANGO_SETTINGS_MODULE"] = "jarvis_project.settings"
    print("啟動JARVIS AI助手，支援語音輸入及股票分析功能...")
