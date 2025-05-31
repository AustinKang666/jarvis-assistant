"""
檢索器模組
負責根據查詢檢索相關文檔
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from modules.rag.vector_store import VectorStore

#   ~看單元五~~
class Retriever:
    """
    檢索器類，負責檢索相關文檔
    """
    
    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """
        初始化檢索器
        
        Args:
            vector_store: 向量存儲實例
            top_k: 返回的最相似文檔數量
        """
        self.vector_store = vector_store
        self.top_k = top_k
        logging.info(f"初始化檢索器，top_k: {top_k}")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        檢索與查詢相關的文檔
        
        Args:
            query: 查詢文本
            
        Returns:
            相關文檔列表，包含文檔內容和相似度
        """
        try:
            # 使用向量存儲進行相似度搜索
            results = self.vector_store.similarity_search(query, self.top_k)
            logging.info(f"查詢 '{query}' 檢索到 {len(results)} 個相關文檔")
            return results
            
        except Exception as e:
            logging.error(f"檢索文檔時出錯: {e}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """
        獲取查詢的上下文信息
        
        Args:
            query: 查詢文本
            
        Returns:
            格式化的上下文信息
        """
        try:
            # 檢索相關文檔
            results = self.retrieve(query)
            
            if not results:
                logging.warning(f"查詢 '{query}' 沒有檢索到相關文檔")
                return ""
            
            # 格式化上下文
            context = "以下是與您問題相關的資訊：\n\n"
            
            for i, result in enumerate(results, 1):
                document = result["document"]
                similarity = result["similarity"]
                
                # 添加文檔內容
                context += f"段落 {i} [相關度: {similarity:.2f}]:\n"
                context += document["text"].strip()
                context += "\n\n"
                
                # 添加來源信息
                if "metadata" in document and document["metadata"]:
                    metadata = document["metadata"]
                    if "source" in metadata:
                        context += f"來源: {metadata['source']}\n\n"
            
            return context.strip()
            
        except Exception as e:
            logging.error(f"獲取上下文時出錯: {e}")
            return ""
