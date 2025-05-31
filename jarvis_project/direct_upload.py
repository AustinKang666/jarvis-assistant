"""
文件上傳處理模組
提供文件上傳和基本處理功能
"""
import os
import logging
from typing import Tuple, Optional
from pathlib import Path

# 項目根目錄
BASE_DIR = Path(__file__).resolve().parent.parent

# 設置日誌
logging.basicConfig(level=logging.INFO)

def handle_uploaded_file(file, is_image: bool = False) -> Tuple[bool, str]:
    """
    處理上傳的文件
    
    Args:
        file: 上傳的文件對象
        is_image: 是否為圖像文件
        
    Returns:
        (成功標誌, 文件路徑或錯誤消息)
    """
    try:
        # 獲取文件名和擴展名
        file_name = file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 檢查文件類型
        if is_image:
            supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
            if file_ext not in supported_formats:
                return False, f"不支持的圖像格式，請上傳 {', '.join(supported_formats)} 格式的圖像"
                
            # 設置圖像保存路徑
            target_dir = os.path.join(BASE_DIR, "data", "images")
        else:
            supported_formats = ['.pdf', '.txt', '.docx', '.doc']
            if file_ext not in supported_formats:
                return False, f"不支持的文件格式，請上傳 {', '.join(supported_formats)} 格式的文件"
                
            # 設置文檔保存路徑
            target_dir = os.path.join(BASE_DIR, "data", "knowledge_base")
        
        # 確保目標目錄存在
        os.makedirs(target_dir, exist_ok=True)
        
        # 生成安全的文件名
        import re
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', file_name)
        file_path = os.path.join(target_dir, safe_filename)
        
        # 保存文件
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
                
        logging.info(f"文件已保存到: {file_path}")
        return True, file_path
        
    except Exception as e:
        logging.error(f"處理上傳文件時出錯: {e}")
        return False, str(e)
