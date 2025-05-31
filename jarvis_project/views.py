"""
Django視圖模組
提供JARVIS AI助理的API端點
"""
import os
import sys
import json
import logging
from datetime import datetime
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import tempfile
import shutil

# 導入文件上傳處理模块
from jarvis_project.direct_upload import handle_uploaded_file

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config
from modules.llm.openai_client import OpenAIClient
from modules.rag.rag_manager import RAGManager
from modules.cache.response_cache import ResponseCache
from modules.safety.content_filter import ContentFilter
from modules.speech.speech_service import SpeechService
from modules.vision.image_analyzer import ImageAnalyzer
from modules.stock.stock_service import StockService

# 初始化OpenAI客戶端
openai_client = OpenAIClient()

# 初始化RAG管理器
rag_manager = RAGManager()

# 初始化回應快取
response_cache = ResponseCache()

# 初始化內容過濾器
content_filter = ContentFilter(
    filter_level=config.SAFETY_FILTER_LEVEL,
    enabled=config.SAFETY_FILTER_ENABLED
)

# 初始化語音服務
speech_service = SpeechService()

# 初始化圖像分析器
image_analyzer = ImageAnalyzer()

# 初始化股票服務
stock_service = StockService()

# 系統提示
SYSTEM_PROMPT = """你是JARVIS，一個功能強大的AI助手。
你能夠幫助用戶回答問題、查找信息、提供建議等。
盡可能提供有用、簡潔但全面的回答。
使用繁體中文回應用戶的問題。
"""

@csrf_exempt
def index(request):
    """首頁視圖"""
    return JsonResponse({"status": "success", "message": "JARVIS AI Assistant API is running"})

@csrf_exempt
def health_check(request):
    """健康檢查端點"""
    return JsonResponse({"status": "healthy"})

@csrf_exempt
def jarvis_api(request):
    """JARVIS API端點"""
    if request.method == 'POST':
        try:
            # 解析請求數據
            data = json.loads(request.body)
            message = data.get('message', '')
            conversation_id = data.get('conversation_id', '')
            message_history = data.get('message_history', [])
            
            # 檢查是否需要使用RAG、網路搜索和快取
            use_rag = data.get('use_rag', True)
            use_web_search = data.get('use_web_search', False)
            use_cache = data.get('use_cache', True)
            use_safety = data.get('use_safety', True)
            safety_level = data.get('safety_level', 'medium')
            
            # 設置安全過濾器級別
            if use_safety and safety_level != content_filter.filter_level:
                content_filter.set_filter_level(safety_level)
            
            # 如果啟用快取，先檢查快取
            if use_cache:
                cached_response = response_cache.get_response(message)
                if cached_response:
                    response_text = cached_response['response']
                    
                    # 如果使用了相似度匹配，添加相關提示
                    if 'similarity' in cached_response:
                        similarity = cached_response.get('similarity', 0.0)
                        response_text += f"\n\n(回答來自快取 - 基於相似問題，相似度: {similarity:.2f})"                        
                    else:
                        response_text += "\n\n(回答來自快取)"
                    
                    # 更新快取統計
                    response_cache.update_stats(message)

                    # 將原始用戶消息添加到歷史
                    message_history.append({"role": "user", "content": message})
                    
                    # 將助手回應添加到歷史
                    message_history.append({"role": "assistant", "content": response_text})
                    
                    # 返回響應
                    response = {
                        "status": "success",
                        "message": response_text,
                        "conversation_id": conversation_id,
                        "message_history": message_history,
                        "from_cache": True
                    }
                    
                    return JsonResponse(response)
            
            
            # 使用OpenAI生成回應
            if use_rag:
                # 使用RAG獲取帶有上下文的提示，可選擇性地使用網路搜索
                enhanced_message = rag_manager.get_prompt_with_context(message, use_web_search=use_web_search)
                
                # 如果獲取到了增強的提示（不等同於原始消息）
                if enhanced_message != message:
                    logging.info(f"使用增強的提示: \n{enhanced_message[:300]}...") # 輸出第一部分作為日誌
                    
                    # 將原始用戶消息添加到歷史
                    message_history.append({"role": "user", "content": message})
                    
                    # 生成回應
                    system_prompt_with_rag = SYSTEM_PROMPT + "\n使用我提供的資料來回答問題。"
                    response_text = openai_client.generate_response([{"role": "user", "content": enhanced_message}], system_prompt_with_rag)
                    
                    # 添加來源註明
                    if use_web_search and "從網路搜索找到的資訊" in enhanced_message:
                        response_text += "\n\n(回答參考了知識庫資料及網路搜索結果)"
                    else:
                        response_text += "\n\n(回答參考了您上傳的知識庫資料)"
                else:
                    # 如果沒有增強提示，使用普通對話模式
                    message_history.append({"role": "user", "content": message})
                    response_text = openai_client.generate_response(message_history, SYSTEM_PROMPT)
            else:
                # 不使用RAG，直接將用戶消息添加到歷史
                message_history.append({"role": "user", "content": message})
                
                # 生成回應
                response_text = openai_client.generate_response(message_history, SYSTEM_PROMPT)
            
            # 如果啟用內容過濾，對回應進行過濾
            if use_safety:
                filter_result = content_filter.filter_content(response_text, context=message)
                
                # 如果內容不安全，使用過濾後的回應
                if not filter_result.get("safe", True):
                    logging.warning(f"檢測到不安全內容，已過濾: {filter_result.get('details', {}).get('category', 'unknown')}")
                    response_text = filter_result.get("filtered", response_text)
                    
                    # 添加過濾提示
                    response_text += "\n\n(該回應已經過安全審核調整)"
                    
                    # 不將不安全的回應添加到快取
                    use_cache = False
            
            # 將助手回應添加到歷史
            message_history.append({"role": "assistant", "content": response_text})
            
            # 如果啟用快取且內容安全，將回應添加到快取
            if use_cache:
                # 確定回應來源類型
                if use_web_search and enhanced_message != message and "從網路搜索找到的資訊" in enhanced_message:
                    source_type = "web_search"
                elif use_rag and enhanced_message != message:
                    source_type = "rag"
                else:
                    source_type = "direct"
                
                # 將回應添加到快取
                response_cache.add_response(
                    question=message,
                    response=response_text,
                    source_type=source_type,
                    metadata={
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "use_rag": use_rag,
                        "use_web_search": use_web_search
                    }
                )
            
            # 返回響應
            response = {
                "status": "success",
                "message": response_text,
                "conversation_id": conversation_id,
                "message_history": message_history
            }
            
            return JsonResponse(response)
        
        except Exception as e:
            logging.error(f"處理請求時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)



@csrf_exempt
def analyze_stock(request):
    """
    🔹 股票分析 API 端點（POST）
    🔹 接收前端傳來的股票代碼（stock_symbol），呼叫 StockService 的 LLM agent 進行完整分析，
       包含股價、公司資訊、財務指標、分析師推薦、新聞與產業比較，並回傳投資建議。
    Returns:
        JsonResponse: 分析結果或錯誤訊息
           - 成功：{"status": "success", "symbol": ..., "analysis": ...}
           - 失敗：{"status": "error", "message": ...}
    """
    # 僅允許 POST 方法
    if request.method == 'POST':
        try:
            # 解析請求數據
            data = json.loads(request.body) # 解析前端發送的 JSON 請求體（request.body 為 bytes 類型）
            stock_symbol = data.get('stock_symbol', '') # 嘗試取得參數 stock_symbol，若缺失則設為空字串（避免拋出錯誤）
                                                        # 配合 app.py內的 async def analyze_stock() => requests.post() 內的 json={"stock_symbol": stock_symbol}
            
            # 若使用者未提供股票代碼，立即回傳錯誤訊息，避免後續處理
            if not stock_symbol:
                logging.warning("使用者未提供股票代碼")
                return JsonResponse({"status": "error", "message": "未提供股票代碼"}, status=400)
            
            # 若使用者有提供股票代碼(台股 or 美股) 
            # => 前端 app.py 內的 async def on_message 已經處理好 stock_symbol 了，因此可直接丟入下述 stock_service.analyze_stock()中
            #                                                                    [app.py 內的 async def analyze_stock 會呼叫此 def]
            # ✅ 呼叫 stock_service.analyze_stock()，會回傳 Tuple[bool, str]  => stock / stock_service.py 內的 def analyze_stock()
            #    - 若成功，success 為 True，result [為分析建議（自然語言）]
            #    - 若失敗，success 為 False，result [為錯誤訊息（如代碼無效、分析中斷）]
            success, result = stock_service.analyze_stock(stock_symbol)

            # 若分析成功，組成成功的 JSON 回傳，狀態碼為 200（預設）
            # ★只有"成功"時才會回傳 status=200 給前端!!!
            if success:
                return JsonResponse({
                    "status": "success",  # 表示分析成功 (前端可根據 status 判斷是否為成功回應)
                    "symbol": stock_symbol, # 回傳使用者查詢的股票代碼，方便前端標示與記錄
                    "analysis": result  #  LLM 回傳的分析結論（整合工具結果、自然語言格式建議）
                })
            
            # 若分析失敗 success == False（如輸入不正確、agent 執行錯誤），回傳錯誤訊息與 400 狀態碼
            else:
                return JsonResponse({
                    "status": "error",  # 統一格式標示錯誤情況
                    "message": result  # 提供詳細錯誤原因（方便前端或使用者顯示）
                }, status=400)

        # ⚠️ 若程式執行過程中發生未預期的例外（如 agent.chat 發生中斷、模組未初始化）
        #    則記錄到日誌中並回傳 HTTP 500（伺服器錯誤）
        except Exception as e:
            logging.error(f"股票分析時出錯（代碼：{stock_symbol}）: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    # ⚠️ 若使用非 POST 方法（如 GET），則明確回傳 405（Method Not Allowed）錯誤
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)



@csrf_exempt
def handle_image_upload(file):
    """處理圖像上傳並進行分析"""
    try:
        # 使用 direct_upload 模塊處理圖像上傳
        success, result = handle_uploaded_file(file, is_image=True)
        
        if not success:
            return JsonResponse({
                "status": "error",
                "message": result
            }, status=400)
            
        image_path = result
        
        # 使用圖像分析器分析圖像
        logging.info(f"開始分析圖像: {image_path}")
        analysis_result = image_analyzer.analyze_image(image_path)
        
        if not analysis_result["success"]:
            return JsonResponse({
                "status": "error",
                "message": f"分析圖像時出错: {analysis_result.get('error', '')}"
            }, status=400)
        
        # 返回分析結果
        return JsonResponse({
            "status": "success",
            "message": f"圖像 {file.name} 已成功分析",
            "analysis": analysis_result["analysis"],
            "file_path": image_path
        })
    except Exception as e:
        logging.error(f"處理圖像上傳時出錯: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)

@csrf_exempt
def analyze_image(request):
    """圖像分析API端點"""
    if request.method == 'POST':
        try:
            # 檢查是否有圖像文件
            if 'file' not in request.FILES and 'image' not in request.FILES:
                return JsonResponse({"status": "error", "message": "請提供圖像文件"}, status=400)
            
            # 獲取文件，支持 'file' 或 'image' 參數名
            image_file = request.FILES.get('file') or request.FILES.get('image')
            
            # 處理圖像上傳
            return handle_image_upload(image_file)
                
        except Exception as e:
            logging.error(f"圖像分析出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)


@csrf_exempt
def upload_file(request):
    """上傳文件端點"""
    if request.method == 'POST':
        try:
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({"status": "error", "message": "未提供文件"}, status=400)
            
            # 獲取文件名和擴展名
            file_name = file.name
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # 檢查是否為圖像文件
            is_image = file_ext in config.VISION_CONFIG.get("supported_formats", ['.jpg', '.jpeg', '.png', '.webp', '.gif'])
            
            # 如果是圖像，呼叫圖像分析函數
            if is_image:
                return handle_image_upload(file)
            
            # 檢查文檔類型
            if file_ext not in ['.pdf', '.txt', '.docx', '.doc']:
                return JsonResponse({
                    "status": "error", 
                    "message": "不支持的文件類型，請上傳PDF、TXT或Word文件，或支持的圖像格式"
                }, status=400)
            
            # 處理文件上傳
            success, result = handle_uploaded_file(file, is_image=False)
            
            if not success:
                return JsonResponse({
                    "status": "error",
                    "message": result
                }, status=400)
                
            file_path = result
            
            # 嘗試將文件添加到知識庫
            try:
                success = rag_manager.add_document(file_path)
                if success:
                    return JsonResponse({
                        "status": "success",
                        "message": f"文件 {file_name} 已成功上傳並添加到知識庫",
                        "file_path": file_path
                    })
                else:
                    return JsonResponse({
                        "status": "partial_success",
                        "message": f"文件 {file_name} 已上傳，但添加到知識庫時出錯"
                    })
            except Exception as e:
                logging.error(f"將文件添加到知識庫時出錯: {e}")
                return JsonResponse({
                    "status": "partial_success",
                    "message": f"文件 {file_name} 已上傳，但添加到知識庫時出錯: {str(e)}"
                })
            
        except Exception as e:
            logging.error(f"處理文件上傳時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)

@csrf_exempt
def rebuild_knowledge_base(request):
    """重建知識庫端點"""
    if request.method == 'POST':
        try:
            # 創建新的RAG管理器
            global rag_manager
            rag_manager = RAGManager()
            
            # 從知識庫目錄添加所有文檔
            count = rag_manager.add_documents_from_directory()
            
            return JsonResponse({
                "status": "success",
                "message": f"知識庫已重建，已添加 {count} 個文檔塊"
            })
        except Exception as e:
            logging.error(f"重建知識庫時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)

@csrf_exempt
def cache_stats(request):
    """快取統計端點"""
    if request.method == 'GET':
        try:
            # 獲取快取統計
            stats = response_cache.get_cache_stats()
            
            return JsonResponse({
                "status": "success",
                "stats": stats
            })
        except Exception as e:
            logging.error(f"獲取快取統計時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援GET請求"}, status=405)

@csrf_exempt
def clear_cache(request):
    """清除快取端點"""
    if request.method == 'POST':
        try:
            # 清除快取
            response_cache.clear_cache()
            
            return JsonResponse({
                "status": "success",
                "message": "快取已成功清除"
            })
        except Exception as e:
            logging.error(f"清除快取時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)

# ~看單元8~ => def content_filter_settings()
@csrf_exempt
def safety_config(request):
    """安全過濾器設置端點"""
    if request.method == 'GET':
        # 獲取安全過濾器的當前設置
        return JsonResponse({
            "status": "success", 
            "safety_config": {
                "enabled": content_filter.enabled,
                "filter_level": content_filter.filter_level,
                "model": content_filter.model
            }
        })
    elif request.method == 'POST':
        try:
            # 解析請求數據
            data = json.loads(request.body)
            
            # 更新過濾級別
            if 'filter_level' in data:
                success = content_filter.set_filter_level(data['filter_level'])
                if not success:
                    return JsonResponse({
                        "status": "error",
                        "message": f"無效的過濾級別: {data['filter_level']}"
                    }, status=400)
            
            # 啟用或停用過濾器
            if 'enabled' in data:
                if data['enabled']:
                    content_filter.enable()
                else:
                    content_filter.disable()
            
            # 返回更新後的設置
            return JsonResponse({
                "status": "success",
                "message": "安全過濾器設置已更新",
                "safety_config": {
                    "enabled": content_filter.enabled,
                    "filter_level": content_filter.filter_level,
                    "model": content_filter.model
                }
            })
        except Exception as e:
            logging.error(f"更新安全過濾器設置時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "僅支援GET和POST請求"}, status=405)

# ~看單元8~ => def content_filter_settings()
@csrf_exempt
def test_safety_filter(request):
    """測試安全過濾器端點"""
    if request.method == 'POST':
        try:
            # 解析請求數據
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return JsonResponse({
                    "status": "error",
                    "message": "未提供測試文本"
                }, status=400)
            
            # 使用安全過濾器測試文本
            filter_result = content_filter.filter_content(text)
            
            # 返回結果
            return JsonResponse({
                "status": "success",
                "result": filter_result
            })
        except Exception as e:
            logging.error(f"測試安全過濾器時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)

@csrf_exempt
def speech_to_text(request):
    """語音轉文字端點"""
    if request.method == 'POST':
        try:
            # 檢查是否有音頻文件
            if 'audio' not in request.FILES:
                return JsonResponse({
                    "status": "error",
                    "message": "未提供音頻文件"
                }, status=400)
            
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
            
            # 檢查音頻是否為空
            if not audio_data:
                return JsonResponse({
                    "status": "error",
                    "message": "音頻文件為空"
                }, status=400)
            
            # 使用語音服務轉換音頻
            result = speech_service.speech_to_text(audio_data)
            
            if result["status"] == "success":
                return JsonResponse({
                    "status": "success",
                    "text": result["text"]
                })
            else:
                return JsonResponse({
                    "status": "error",
                    "message": result.get("error", "語音轉文字失敗")
                }, status=500)
                
        except Exception as e:
            logging.error(f"處理語音轉文字請求時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)

@csrf_exempt
def text_to_speech(request):
    """文字轉語音端點"""
    if request.method == 'POST':
        try:
            # 解析請求數據
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return JsonResponse({
                    "status": "error",
                    "message": "未提供文字內容"
                }, status=400)
            
            # 使用ElevenLabs或默認OpenAI
            use_elevenlabs = data.get('use_elevenlabs', False)
            voice_id = data.get('voice_id', None)
            
            # 使用語音服務生成音頻
            try:
                audio_data, mime_type = speech_service.text_to_speech(
                    text, 
                    voice_id=voice_id,
                    use_elevenlabs=use_elevenlabs
                )
                
                # 創建HTTP響應
                response = HttpResponse(audio_data, content_type=mime_type)
                response['Content-Disposition'] = 'attachment; filename="speech.mp3"'
                return response
                
            except Exception as e:
                logging.error(f"生成語音時出錯: {e}")
                return JsonResponse({
                    "status": "error",
                    "message": f"生成語音時出錯: {str(e)}"
                }, status=500)
                
        except Exception as e:
            logging.error(f"處理文字轉語音請求時出錯: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({"status": "error", "message": "只支援POST請求"}, status=405)
