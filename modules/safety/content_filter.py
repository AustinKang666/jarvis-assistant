"""
內容安全過濾模組
使用 GPT-4o-mini 作為內容審查守衛，過濾不適當內容
"""
import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from modules.llm.openai_client import OpenAIClient

# ~可看單元8~
class ContentFilter:
    """
    內容安全過濾器類
    使用 GPT-4o-mini 判斷內容是否適當並提供替代回應
    """
    
    def __init__(self, filter_level: str = "medium", enabled: bool = True):
        """
        初始化內容過濾器
        
        Args:
            filter_level: 過濾級別 ("low", "medium", "high")
            enabled: 是否啟用過濾器
        """
        self.filter_level = filter_level
        self.enabled = enabled
        self.openai_client = OpenAIClient()
        self.model = "gpt-4o-mini"  # 使用 gpt-4o-mini 進行內容審核
        
        logging.info(f"初始化內容安全過濾器，級別: {filter_level}, 狀態: {'啟用' if enabled else '停用'}")
        
        # 安全守衛的 system prompt
        self.system_prompt = """你是 JARVIS 的內容安全審查員，負責確保所有回應不包含不適當或有害內容。
        
你的任務是識別並過濾以下類型的不適當內容：

1. 暴力、虐待或仇恨言論：包括對個人或群體的暴力描述、威脅或仇恨言論
2. 色情或過度露骨的性內容：包括露骨的性描述、性暗示或性別歧視內容
3. 自我傷害或自殺相關內容：包括鼓勵、美化或提供自我傷害方法的內容
4. 非法活動指引：包括製作武器、毒品、破解系統等違法行為的具體指導
5. 用戶資料隱私風險：包含可能誤導用戶分享個人敏感資料的內容
6. 欺詐或惡意操縱：包括詐騙、惡意社會工程學等欺騙用戶的內容
7. 潛在偏見或歧視性言論：包括對特定族群、性別、宗教等的偏見或歧視
8. 不實或誤導性醫療建議：未經專業醫療認證的健康或醫療建議
9. 政治敏感或極端政治內容：涉及極端政治立場、煽動性政治言論
10. 財務風險建議：未經專業認證的財務投資建議或高風險行為

審查標準根據過濾級別調整：
- low (低): 只過濾明確有害、暴力、色情或違法內容
- medium (中): 過濾有害內容及可能有爭議的敏感內容
- high (高): 最嚴格標準，過濾所有邊界情況和敏感話題

你必須以JSON格式回應，無需任何額外說明或前言。格式如下：
{
    "safe": true/false,
    "category": "none" 或上述類別之一,
    "confidence": 0-1之間的數值,
    "reason": "簡短說明為何此內容被標記或安全",
    "suggestion": "如何修改使內容更安全(如適用)"
}

保持客觀，避免過度審查。你的唯一職責是識別不適當內容，而非審查觀點或事實陳述。
"""
    
    def filter_content(self, content: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        過濾內容，判斷是否適當
        
        Args:
            content: 需要審查的內容
            context: 相關上下文（可選）
            
        Returns:
            包含過濾結果的字典
        """
        # 如果過濾器已停用，則直接返回內容為安全
        if not self.enabled:
            logging.info("內容過濾器已停用，跳過審查")
            return {
                "original": content,
                "safe": True,
                "filtered": content,
                "details": {"message": "content filtering disabled"}
            }
        
        logging.info(f"正在使用 {self.model} 審查內容，過濾級別: {self.filter_level}")
        
        # 準備審查內容的提示
        prompt = f"""
        內容審查請求:
        ---
        過濾級別: {self.filter_level}

        需要審查的內容:
        \"\"\"{content}\"\"\"
        """
        
        if context:
            prompt += f"""
            相關上下文資訊:
            \"\"\"{context}\"\"\"
            """
            
        prompt += """
        請評估上述內容是否安全適當，並以JSON格式回應。
        """
        
        try:
            # 調用 OpenAI API 進行審查
            # 使用 generate_response 方法而非 generate_text
            response = self.openai_client.generate_response(
                [{"role": "user", "content": prompt}], 
                system_prompt=self.system_prompt
            )
            
            # 嘗試解析 JSON 回應
            try:
                result = json.loads(response)
                logging.info(f"審查結果: safe={result.get('safe', True)}, category={result.get('category', 'none')}, confidence={result.get('confidence', 0)}")
                
                if not result.get("safe", True):
                    # 如果內容不安全，提供替代回應
                    safe_response = self._create_safe_response(
                        content, 
                        result.get("category", "unknown"), 
                        result.get("suggestion", "")
                    )
                    return {
                        "original": content,
                        "safe": False,
                        "filtered": safe_response,
                        "details": result
                    }
                else:
                    # 內容安全，直接通過
                    return {
                        "original": content,
                        "safe": True,
                        "filtered": content,
                        "details": result
                    }
            except json.JSONDecodeError as e:
                # JSON 解析錯誤
                logging.error(f"無法解析審查結果為 JSON: {e}")
                logging.debug(f"原始回應: {response}")
                
                # 保守處理，判斷是否有明確安全/不安全指示
                if "不安全" in response or "有害" in response or "inappropriate" in response:
                    return {
                        "original": content,
                        "safe": False,
                        "filtered": self._create_safe_response(content, "unknown", ""),
                        "details": {"error": "審查回應解析失敗但有風險指示"}
                    }
                else:
                    # 無法判斷，視為安全
                    return {
                        "original": content,
                        "safe": True,
                        "filtered": content,
                        "details": {"error": "審查回應解析失敗"}
                    }
                    
        except Exception as e:
            # 審查過程出錯
            logging.error(f"內容審查過程出錯: {e}")
            # 保守處理，仍讓內容通過
            return {
                "original": content,
                "safe": True,
                "filtered": content,
                "details": {"error": f"審查過程出錯: {str(e)}"}
            }
    
    def _create_safe_response(self, original: str, category: str, suggestion: str) -> str:
        """
        根據不同類別的問題，提供適當的替代回應
        
        Args:
            original: 原始內容
            category: 問題類別
            suggestion: 建議修改方向
            
        Returns:
            安全的替代回應
        """
        # 根據不同類別提供不同回應
        category_responses = {
            "violence": "我無法提供涉及暴力內容的回應。作為AI助理，我致力於提供安全且有幫助的資訊。請問有其他我可以協助您的事情嗎？",
            "hate_speech": "我無法提供可能包含仇恨言論的內容。我致力於尊重和包容所有人群。請問您有其他問題需要協助嗎？",
            "sexual": "很抱歉，我無法提供涉及不適當或成人內容的回應。請問有其他我可以協助您的問題嗎？",
            "self_harm": "您的問題涉及自我傷害相關內容，我建議您尋求專業人士的協助。若您正在經歷危機，請立即聯絡心理健康專線或緊急服務。",
            "illegal": "很抱歉，我無法提供可能涉及違法活動的指導或建議。我只能提供合法且安全的資訊。",
            "privacy_risk": "您的問題可能涉及隱私風險，我建議不要在線上分享個人敏感資訊，以保護您的安全。",
            "fraud": "我無法提供可能涉及欺詐或操縱行為的內容。作為AI助理，我致力於提供真實且有益的資訊。",
            "bias": "您的問題可能涉及偏見或歧視性內容。我致力於公平對待所有人群，並避免傳播可能造成傷害的偏見。",
            "medical": "我無法提供醫療建議，這類問題應該由專業醫療人員回答。若您有健康相關疑問，請諮詢合格的醫生或醫療專業人員。",
            "political": "您的問題涉及敏感政治內容，我無法提供可能引起爭議或煽動性的政治觀點。我可以提供更中立的資訊或討論其他主題。",
            "financial_risk": "我無法提供可能涉及財務風險的投資建議。財務決策應諮詢專業金融顧問，並考慮您的個人財務狀況。"
        }
        
        # 獲取對應類別的回應，或使用通用回應
        response = category_responses.get(
            category, 
            "很抱歉，我無法提供這類內容。作為AI助理，我致力於提供安全、有益且適當的資訊。請問我可以以其他方式幫助您嗎？"
        )
        
        # 如果有具體建議且過濾級別不是高，嘗試提供修改後的建議
        if suggestion and self.filter_level != "high":
            response += f"\n\n我可以嘗試以更適當的方式回答您的問題。{suggestion}"
        
        return response
    
    def set_filter_level(self, level: str) -> bool:
        """
        設置過濾級別
        
        Args:
            level: 新的過濾級別 ("low", "medium", "high")
            
        Returns:
            是否成功設置
        """
        if level.lower() in ["low", "medium", "high"]:
            self.filter_level = level.lower()
            logging.info(f"內容過濾級別已設置為: {self.filter_level}")
            return True
        else:
            logging.warning(f"無效的過濾級別: {level}")
            return False
    
    def enable(self) -> None:
        """啟用內容過濾"""
        self.enabled = True
        logging.info("內容過濾已啟用")
    
    def disable(self) -> None:
        """停用內容過濾"""
        self.enabled = False
        logging.info("內容過濾已停用")
    
    def is_enabled(self) -> bool:
        """
        檢查過濾器是否啟用
        
        Returns:
            是否啟用
        """
        return self.enabled
    
    def get_status(self) -> Dict[str, Any]:
        """
        獲取過濾器狀態
        
        Returns:
            包含過濾器狀態的字典
        """
        return {
            "enabled": self.enabled,
            "filter_level": self.filter_level,
            "model": self.model
        }
