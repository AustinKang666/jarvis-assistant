"""
向量存儲模組
負責將文檔嵌入為向量並提供檢索功能
"""
import os
import sys
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

#  ~看單元五~~
class VectorStore:
    """
    向量存儲類，負責管理文檔嵌入
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量存儲
        
        Args:
            model_name: 嵌入模型名稱
        """
        self.documents = []  # 存儲文檔
        self.vectors = []    # 存儲向量
        
        # 加載嵌入模型
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"已加載嵌入模型: {model_name}")
        except Exception as e:
            logging.error(f"加載嵌入模型時出錯: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加文檔到向量存儲
        
        Args:
            documents: 文檔列表，每個文檔是包含 text 和 metadata 的字典
        """
        if not documents:
            logging.warning("沒有文檔需要添加")
            return
        
        try:
            # 獲取文本列表
            texts = [doc["text"] for doc in documents]
            
            # 生成嵌入向量
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # 添加到存儲
            self.documents.extend(documents)
            self.vectors.extend(embeddings)
            
            logging.info(f"已添加 {len(documents)} 個文檔到向量存儲")
            
        except Exception as e:
            logging.error(f"添加文檔時出錯: {e}")
            raise
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        基於相似度搜索文檔
        
        Args:
            query: 查詢文本
            top_k: 返回的最相似文檔數量
            
        Returns:
            相似文檔列表，包含文檔內容和相似度
        """
        if not self.documents:
            logging.warning("向量存儲為空，無法執行搜索")
            return []
        
        if top_k <= 0:
            logging.warning(f"無效的 top_k 值: {top_k}，使用默認值 3")
            top_k = 3
        
        try:
            # 為查詢生成嵌入向量
            query_vector = self.model.encode(query, convert_to_tensor=False)
            
            # 計算相似度
            similarities = self._calculate_similarities(query_vector)
            
            # 獲取最相似的文檔索引
            if len(similarities) <= top_k:
                top_indices = list(range(len(similarities)))
            else:
                top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # 整理結果
            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx],
                    "similarity": similarities[idx]
                })
            
            return results
            
        except Exception as e:
            logging.error(f"搜索文檔時出錯: {e}")
            return []
    
    def _calculate_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """
        計算查詢向量與所有文檔向量的相似度
        
        Args:
            query_vector: 查詢向量
            
        Returns:
            相似度數組
        """
        # 將列表轉為 NumPy 數組
        vectors_array = np.array(self.vectors)
        
        # 向量標準化
        vectors_array = vectors_array / np.linalg.norm(vectors_array, axis=1, keepdims=True)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 計算餘弦相似度
        similarities = np.dot(vectors_array, query_vector)
        
        return similarities
    
    def save(self, file_path: str) -> bool:
        """
        保存向量存儲到文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            是否成功保存
        """
        try:
            data = {
                "documents": self.documents,
                "vectors": self.vectors
            }
            
            # 保存數據
            with open(f"{file_path}.docs", "wb") as f_docs:
                pickle.dump(data["documents"], f_docs)
                
            with open(f"{file_path}.index", "wb") as f_index:
                pickle.dump(data["vectors"], f_index)
            
            logging.info(f"已保存向量存儲到: {file_path}.docs 和 {file_path}.index")
            return True
            
        except Exception as e:
            logging.error(f"保存向量存儲時出錯: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> "VectorStore":
        """
        從文件加載向量存儲
        
        Args:
            file_path: 文件路徑
            
        Returns:
            加載的向量存儲實例
        """
        try:
            # 實例化類
            instance = cls()
            
            # 加載數據
            with open(f"{file_path}.docs", "rb") as f_docs:
                instance.documents = pickle.load(f_docs)
                
            with open(f"{file_path}.index", "rb") as f_index:
                instance.vectors = pickle.load(f_index)
            
            logging.info(f"已加載向量存儲，包含 {len(instance.documents)} 個文檔")
            return instance
            
        except Exception as e:
            logging.error(f"加載向量存儲時出錯: {e}")
            raise
