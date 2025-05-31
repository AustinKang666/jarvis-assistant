"""
圖像分析模組
使用 GPT-4o-mini 進行圖像理解與分析
"""
import os
import sys
import base64
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import io

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from modules.llm.openai_client import OpenAIClient

class ImageAnalyzer:
    """
    圖像分析類
    使用 OpenAI 的 GPT-4o-mini 進行視覺理解
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化圖像分析器
        
        Args:
            api_key: OpenAI API 密鑰，如果不提供則從配置文件獲取
        """
        self.openai_client = OpenAIClient(api_key=api_key)
        self.model = config.VISION_CONFIG.get("model", "gpt-4o")  # 使用支援視覺功能的模型
        logging.info("圖像分析器已初始化")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        將圖像編碼為 base64 字符串
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            base64 編碼的圖像字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"圖像編碼錯誤: {e}")
            raise
    
    def resize_image_if_needed(self, image_path: str, max_size: int = 4 * 1024 * 1024) -> str:
        """
        如果圖像太大，將其調整大小
        
        Args:
            image_path: 圖像文件路徑
            max_size: 最大文件大小 (字節)
            
        Returns:
            可能調整大小後的圖像路徑
        """
        try:
            # 檢查文件大小
            file_size = os.path.getsize(image_path)
            if file_size <= max_size:
                return image_path
                
            logging.info(f"圖像大小 ({file_size} 字節) 超過限制，調整大小")
            
            # 打開並調整圖像大小
            img = Image.open(image_path)
            
            # 計算調整比例
            ratio = (max_size / file_size) ** 0.5
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            
            # 調整大小
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保存調整後的圖像
            filename, ext = os.path.splitext(image_path)
            resized_path = f"{filename}_resized{ext}"
            resized_img.save(resized_path, quality=85)
            
            logging.info(f"圖像已調整大小並保存到: {resized_path}")
            return resized_path
        except Exception as e:
            logging.error(f"調整圖像大小時出錯: {e}")
            return image_path  # 返回原始路徑
    
    def analyze_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        分析圖像內容
        
        Args:
            image_path: 圖像文件路徑
            prompt: 自定義提示文本 (可選)
            
        Returns:
            包含分析結果的字典
        """
        try:
            logging.info(f"開始分析圖像: {image_path}")
            
            # 如果未提供提示，使用默認提示
            if not prompt:
                prompt = "請詳細描述這張圖像中的內容。請直接描述你看到的內容，不要使用「這張圖片顯示」等詞語。請具體說明圖中的物體、人物、場景或文字。"
            
            # 檢查文件是否存在
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"找不到圖像文件: {image_path}"
                }
            
            # 必要時調整圖像大小
            resized_path = self.resize_image_if_needed(image_path)
            
            # 將圖像編碼為 base64
            base64_image = self.encode_image_to_base64(resized_path)
            
            # 如果調整了大小，刪除臨時文件
            if resized_path != image_path:
                os.remove(resized_path)
            
            # 構建請求消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 調用 OpenAI API 分析圖像
            try:
                system_prompt = """你是一個具有視覺能力的AI助手。
對於用戶提供的圖像，請仔細分析其中的所有細節，並提供清晰詳細的描述。
必須直接描述圖像中的內容，不要說「這張圖片顯示」或「我無法查看圖像」等存在疑慮的詞句。
使用繁體中文回答，並提供詳細說明。
如果圖像包含敏感或不適當的內容，請禮貌地拒絕詳細描述，只提供一般性描述。
"""

                response = self.openai_client.generate_vision_response(messages, system_prompt, model=self.model)
            except Exception as e:
                logging.error(f"調用 OpenAI 視覺 API 出錯: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": f"視覺模型回應出錯: {str(e)}"
                }
            
            # 記錄分析完成
            logging.info(f"圖像分析完成: {image_path}")
            
            return {
                "success": True,
                "analysis": response,
                "prompt": prompt
            }
        except Exception as e:
            logging.error(f"圖像分析出錯: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def identify_text_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        識別圖像中的文字
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            包含文字識別結果的字典
        """
        prompt = "請識別並提取此圖像中的所有文字內容。如果有多種語言，請標明並翻譯成中文。"
        return self.analyze_image(image_path, prompt)
    
    def analyze_objects(self, image_path: str) -> Dict[str, Any]:
        """
        分析圖像中的物體
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            包含物體分析結果的字典
        """
        prompt = "請辨識並列出圖像中所有可見的物體，包括它們的大致位置和相對大小。"
        return self.analyze_image(image_path, prompt)
    
    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        """
        分析圖像場景
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            包含場景分析結果的字典
        """
        prompt = "請描述這個場景，包括環境類型、時間、天氣條件（如適用）以及場景的整體氛圍。"
        return self.analyze_image(image_path, prompt)
    
    def analyze_with_custom_prompt(self, image_path: str, custom_prompt: str) -> Dict[str, Any]:
        """
        使用自定義提示分析圖像
        
        Args:
            image_path: 圖像文件路徑
            custom_prompt: 自定義提示
            
        Returns:
            包含自定義分析結果的字典
        """
        return self.analyze_image(image_path, custom_prompt)


# 測試函數
def test_image_analyzer():
    """
    測試圖像分析器功能
    """
    analyzer = ImageAnalyzer()
    
    # 測試圖像路徑
    test_image = os.path.join(project_root, "data", "test", "test_image.jpg")
    
    # 確保測試目錄存在
    os.makedirs(os.path.dirname(test_image), exist_ok=True)
    
    # 如果測試圖像不存在，創建一個簡單的測試圖像
    if not os.path.exists(test_image):
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (300, 200), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "測試圖像", fill=(255, 255, 0))
            img.save(test_image)
            print(f"已創建測試圖像: {test_image}")
        except Exception as e:
            print(f"創建測試圖像時出錯: {e}")
            return
    
    # 測試圖像分析
    result = analyzer.analyze_image(test_image)
    print(f"分析結果: {result}")


if __name__ == "__main__":
    # 執行測試
    test_image_analyzer()
