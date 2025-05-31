import os
import sys
import requests
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

# 加載環境變量
load_dotenv()

# 設置日誌記錄器
logging.basicConfig(level=logging.INFO)

class SearchService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY") or os.getenv("SERPER_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # 加強鍵值確認
        print(f"\n====== 檢查SERPAPI設定 ======")
        print(f"API 鍵值狀態: {self.api_key is not None}")
        
        # 記錄 API 金鑰狀態（沒有顯示實際的金鑰）
        if self.api_key:
            logging.info(f"SerpAPI 金鑰已設置，長度: {len(self.api_key)}")
            logging.info(f"SerpAPI 金鑰前六位: {self.api_key[:6]}...")
        else:
            logging.warning("SerpAPI 金鑰未設置，網路搜索功能將不可用")
            
        # 初始化快取和請求計數
        self.cache = {}
        self.daily_requests = 0
        self.MAX_DAILY_REQUESTS = 100
            
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        執行網路搜索並返回結果
        
        Args:
            query: 搜索查詢
            num_results: 要返回的結果數量
            
        Returns:
            搜索結果列表，每個結果包含標題、描述和 URL
        """
        # 檢查API密鑰
        if not self.api_key:
            logging.error("缺少API密鑰，無法執行搜索")
            return []
            
        # 檢查快取
        cache_key = f"{query}_{num_results}"
        if cache_key in self.cache:
            logging.info(f"使用快取結果: {query}")
            return self.cache[cache_key]
            
        # 檢查請求限制
        if self.daily_requests >= self.MAX_DAILY_REQUESTS:
            logging.warning(f"已達到每日搜索上限: {self.MAX_DAILY_REQUESTS}")
            return []
            
        # 記錄搜索請求
        logging.info(f"執行網路搜索: {query}")
        
        # 確保網路搜索可以正確工作
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "num": num_results * 3,  # 請求更多結果
            "api_key": self.api_key,
            "engine": "google",
            "gl": "tw",  # 單位制地區代碼，載入臺灣結果
            "hl": "zh-tw"  # 語言設定為繁體中文
        }
        
        try:
            logging.info(f"請求 SerpAPI: {url} (查詢: {query})")
            response = requests.get(url, params=params)
            
            # 檢查回應狀態
            logging.info(f"SerpAPI 回應狀態碼: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # 記錄回應的基本信息
            logging.info(f"收到 SerpAPI 回應，包含 {len(data.keys())} 個頂層鍵")
            if "organic_results" not in data:
                logging.warning("SerpAPI 回應中沒有找到 'organic_results'")
                logging.info(f"回應中的鍵: {', '.join(data.keys())}")
            
            # 解析搜索結果
            organic_results = data.get("organic_results", [])
            logging.info(f"找到 {len(organic_results)} 個有機搜索結果")
            
            for item in organic_results[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
            
            # 更新快取和請求計數
            self.cache[cache_key] = results
            self.daily_requests += 1
            
            logging.info(f"搜索結果數量: {len(results)}")
            
            return results
            
        except Exception as e:
            logging.error(f"執行搜索時出錯: {str(e)}")
            return []
    
    def enrich_query_with_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        搜索相關資訊並將結果格式化為 LLM 可用的上下文
        
        Args:
            query: 搜索查詢
            num_results: 要返回的結果數量
            
        Returns:
            包含查詢、上下文和原始結果的字典
        """
        try:
            if not self.api_key:
                return {
                    "query": query,
                    "search_context": "網路搜索功能未啟用，請設置API密鑰。",
                    "raw_results": [],
                    "success": False
                }
                
            logging.info(f"開始為查詢豐富上下文: {query}")
            search_results = self.search(query, num_results)
            
            if not search_results:
                return {
                    "query": query,
                    "search_context": "沒有找到相關的網路搜索結果。",
                    "raw_results": [],
                    "success": False
                }
            
            # 格式化搜索結果為上下文
            context = "從網路搜索找到的資訊:\n\n"
            for idx, result in enumerate(search_results, 1):
                context += f"{idx}. {result['title']}\n"
                context += f"   摘要: {result['snippet']}\n"
                context += f"   連結: {result['link']}\n\n"
            
            logging.info(f"成功獲取並格式化網路搜索結果，上下文長度: {len(context)}")
            
            return {
                "query": query,
                "search_context": context,
                "raw_results": search_results,
                "success": True
            }
        except Exception as e:
            logging.error(f"網路搜索出錯: {str(e)}")
            return {
                "query": query,
                "search_context": f"無法獲取搜索結果: {str(e)}",
                "raw_results": [],
                "success": False
            }
