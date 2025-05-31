"""
OpenAI 客戶端模組
負責與 OpenAI API 的通信和回應生成
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# 加載環境變量
load_dotenv()

# 導入配置
import sys
import os

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

class OpenAIClient:
    """
    OpenAI客戶端類，負責處理與OpenAI API的通信
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化OpenAI客戶端
        
        Args:
            api_key: OpenAI API 密鑰，如果不提供則從環境變量獲取
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error("未設置OpenAI API密鑰，請在.env文件中設置OPENAI_API_KEY")
            raise ValueError("OpenAI API key is required")
        
        # 設置默認模型和參數
        self.default_model = config.OPENAI_MODEL
        self.temperature = float(config.OPENAI_TEMPERATURE)
        self.max_tokens = 2000  # 默認最大輸出令牌數
        
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(api_key=self.api_key)
        
        logging.info(f"OpenAI客戶端已初始化，使用模型: {self.default_model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, Any]], 
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = None,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """
        生成AI回應
        
        Args:
            messages: 消息歷史列表
            system_prompt: 系統提示（可選）
            model: 模型名稱（可選，默認使用配置中的模型）
            temperature: 溫度參數（可選，默認使用配置中的值）
            max_tokens: 最大令牌數（可選，默認使用配置中的值）
            
        Returns:
            生成的回應文本
        """
        try:
            # 準備請求消息
            request_messages = []
            
            # 添加系統提示（如果提供）
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            
            # 添加消息歷史
            request_messages.extend(messages)
            
            # 記錄請求信息
            logging.info(f"OpenAI API請求，模型: {model or self.default_model}")
            
            # 調用API
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=request_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            # 提取回應文本
            response_text = response.choices[0].message.content.strip()
            
            return response_text
            
        except Exception as e:
            logging.error(f"OpenAI API請求失敗: {e}")
            # 在生產環境中，可能需要返回一個優雅的錯誤消息
            # 但在開發過程中，讓錯誤傳播可能更有用
            raise
            
    def generate_vision_response(self, 
                                messages: List[Dict[str, Any]], 
                                system_prompt: Optional[str] = None,
                                model: Optional[str] = None) -> str:
        """
        生成帶有視覺理解的回應
        
        Args:
            messages: 包含圖像和文本的訊息列表
            system_prompt: 系統提示
            model: 使用的模型，默認使用視覺模型
            
        Returns:
            生成的回應文本
        """
        try:
            # 使用指定模型或默認視覺模型
            vision_model = model or config.VISION_CONFIG.get("model", self.default_model)
            
            # 準備消息列表
            request_messages = []
            
            # 添加系統提示（如果提供）
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            
            # 添加消息歷史
            request_messages.extend(messages)
            
            # 記錄請求信息
            logging.info(f"OpenAI 視覺API請求，模型: {vision_model}")
            
            # 調用API
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=request_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 提取回應文本
            response_text = response.choices[0].message.content.strip()
            
            return response_text
            
        except Exception as e:
            logging.error(f"OpenAI 視覺API請求失敗: {e}")
            return f"抱歉，我在處理圖像時遇到了問題: {str(e)}"
