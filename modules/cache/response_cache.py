"""
回應快取模組
負責儲存和檢索快取的回應，以提高系統回應效率
"""
import os
import sys
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config

# ~可看單元7~
class ResponseCache:
    """
    回應快取類
    使用向量相似度來尋找相似問題，實現智能快取
    """
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 similarity_threshold: float = 0.85,
                 cache_ttl_days: int = 7):
        """
        初始化回應快取

        Args:
            cache_dir: 快取目錄，如果未指定，使用預設目錄
            embedding_model: 用於生成句子嵌入的模型名稱
            similarity_threshold: 判定相似問題的閾值 (0.0-1.0)
            cache_ttl_days: 快取有效期（天）
        """
        # 設定快取目錄
        self.cache_dir = cache_dir or os.path.join(project_root, "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 設定快取檔案路徑
        self.cache_file = os.path.join(self.cache_dir, "response_cache.json")
        self.vector_file = os.path.join(self.cache_dir, "vectors.npy")
        
        # 設定快取相關參數
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.question_vectors = {}
        
        # 載入快取
        self.cache_data = self._load_cache()
        
        # 初始化句子模型
        try:
            logging.info(f"加載語義模型: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            self.use_semantic = True
        except Exception as e:
            logging.error(f"無法載入句子嵌入模型: {e}, 將退回到基本快取")
            self.use_semantic = False
        
        # 載入向量數據
        if self.use_semantic:
            self._load_vectors()
    
    def _load_cache(self) -> Dict[str, Any]:
        """
        載入快取資料
        
        Returns:
            快取資料字典
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logging.info(f"已載入快取記錄 {len(cache_data)} 條")
                
                # 清理過期快取
                return self._clean_expired_cache(cache_data)
            except Exception as e:
                logging.error(f"載入快取失敗: {e}")
                return {}
        else:
            logging.info("快取檔案不存在，創建新的快取")
            return {}
    
    def _clean_expired_cache(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理過期的快取記錄
        
        Args:
            cache_data: 快取資料字典
            
        Returns:
            清理後的快取資料字典
        """
        current_time = datetime.now()
        cleaned_cache = {}
        
        for key, item in cache_data.items():
            # 檢查時間戳
            timestamp_str = item.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if current_time - timestamp <= self.cache_ttl:
                        cleaned_cache[key] = item
                except ValueError:
                    # 無效的時間戳，保留項目但更新時間戳
                    item['timestamp'] = current_time.isoformat()
                    cleaned_cache[key] = item
            else:
                # 缺少時間戳，添加當前時間
                item['timestamp'] = current_time.isoformat()
                cleaned_cache[key] = item
        
        if len(cache_data) != len(cleaned_cache):
            logging.info(f"已清理 {len(cache_data) - len(cleaned_cache)} 條過期快取記錄")
        
        return cleaned_cache
    
    def _save_cache(self) -> None:
        """保存快取資料到檔案"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logging.info(f"已保存 {len(self.cache_data)} 條快取記錄")
            
            # 保存向量資料
            if self.use_semantic and self.question_vectors:
                self._save_vectors()
        except Exception as e:
            logging.error(f"保存快取失敗: {e}")
    
    def _load_vectors(self) -> None:
        """載入問題向量資料"""
        if os.path.exists(self.vector_file):
            try:
                self.question_vectors = np.load(self.vector_file, allow_pickle=True).item()
                logging.info(f"已載入 {len(self.question_vectors)} 條問題向量")
            except Exception as e:
                logging.error(f"載入向量資料失敗: {e}")
                self.question_vectors = {}
    
    def _save_vectors(self) -> None:
        """保存問題向量資料"""
        try:
            np.save(self.vector_file, self.question_vectors)
            logging.info(f"已保存 {len(self.question_vectors)} 條問題向量")
        except Exception as e:
            logging.error(f"保存向量資料失敗: {e}")
    
    def _get_cache_key(self, question: str) -> str:
        """
        根據問題生成快取鍵
        
        Args:
            question: 使用者問題
            
        Returns:
            快取鍵
        """
        # 使用 MD5 生成快取鍵
        return hashlib.md5(question.encode('utf-8')).hexdigest()
    
    def _get_question_embedding(self, question: str) -> np.ndarray:
        """
        獲取問題的向量嵌入
        
        Args:
            question: 使用者問題
            
        Returns:
            問題的向量嵌入
        """
        if not self.use_semantic:
            return np.array([])
        
        # 檢查是否已經計算過這個問題的向量
        question_hash = self._get_cache_key(question)
        if question_hash in self.question_vectors:
            return self.question_vectors[question_hash]
        
        # 計算問題向量
        try:
            embedding = self.model.encode(question)
            # 儲存計算的向量以便重用
            self.question_vectors[question_hash] = embedding
            return embedding
        except Exception as e:
            logging.error(f"計算問題向量時出錯: {e}")
            return np.array([])
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        計算兩個向量的餘弦相似度
        
        Args:
            vec1: 第一個向量
            vec2: 第二個向量
            
        Returns:
            餘弦相似度 (0.0-1.0)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def find_similar_question(self, question: str) -> Tuple[str, float]:
        """
        尋找與給定問題最相似的已快取問題
        
        Args:
            question: 使用者問題
            
        Returns:
            元組 (最相似的問題, 相似度分數)
        """
        if not self.use_semantic or not self.cache_data:
            return ("", 0.0)
            
        # 取得問題向量
        question_vec = self._get_question_embedding(question)
        if len(question_vec) == 0:
            return ("", 0.0)
            
        # 比較所有已快取問題，找出相似度最高的
        best_match = ""
        best_score = 0.0
        
        for cached_item in self.cache_data.values():
            cached_question = cached_item.get('question', '')
            if not cached_question:
                continue
                
            # 獲取或計算已快取問題的向量
            cached_question_hash = self._get_cache_key(cached_question)
            if cached_question_hash not in self.question_vectors:
                self.question_vectors[cached_question_hash] = self.model.encode(cached_question)
                
            cached_vec = self.question_vectors[cached_question_hash]
            similarity = self._calculate_similarity(question_vec, cached_vec)
            
            if similarity > best_score:
                best_score = similarity
                best_match = cached_question
                
        return (best_match, best_score)
    
    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """
        嘗試從快取中獲取回應
        
        Args:
            question: 使用者問題
            
        Returns:
            包含回應的字典或 None (如果沒有找到快取)
        """
        # 先直接查詢完全匹配
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache_data:
            cached_item = self.cache_data[cache_key]
            logging.info(f"在快取中找到完全匹配: {question[:30]}...")
            # 更新快取訪問時間
            cached_item['last_accessed'] = datetime.now().isoformat()
            self.cache_data[cache_key] = cached_item
            return cached_item
            
        # 如果沒有完全匹配，尋找相似問題
        if self.use_semantic:
            similar_question, similarity = self.find_similar_question(question)
            logging.info(f"相似度最高的問題 ({similarity:.2f}): {similar_question[:30]}...")
            
            if similarity >= self.similarity_threshold:
                # 找到足夠相似的問題
                similar_key = self._get_cache_key(similar_question)
                cached_item = self.cache_data[similar_key]
                logging.info(f"使用相似問題的快取 (相似度: {similarity:.2f})")
                
                # 更新快取訪問時間和相似度記錄
                cached_item['last_accessed'] = datetime.now().isoformat()
                cached_item['similarity_matches'] = cached_item.get('similarity_matches', 0) + 1
                self.cache_data[similar_key] = cached_item
                
                # 返回找到的快取，並添加相似度信息
                result = dict(cached_item)
                result['similarity'] = similarity
                result['original_question'] = similar_question
                return result
                
        # 未找到匹配的快取
        return None
    
    def add_response(self, question: str, response: str, 
                     source_type: str = "direct", metadata: Dict[str, Any] = None) -> None:
        """
        添加回應到快取
        
        Args:
            question: 使用者問題
            response: 系統回應
            source_type: 回應來源類型 ("direct", "rag", "web_search")
            metadata: 附加元數據
        """
        if not question or not response:
            return
            
        # 準備快取項目
        cache_key = self._get_cache_key(question)
        current_time = datetime.now()
        
        cache_item = {
            'question': question,
            'response': response,
            'source_type': source_type,
            'timestamp': current_time.isoformat(),
            'last_accessed': current_time.isoformat(),
            'access_count': 1
        }
        
        # 如果有提供額外元數據，添加到快取
        if metadata:
            cache_item['metadata'] = metadata
            
        # 存儲到快取
        self.cache_data[cache_key] = cache_item
        
        # 如果使用語義搜索，計算並儲存問題向量
        if self.use_semantic:
            self._get_question_embedding(question)
            
        # 保存快取文件
        self._save_cache()
        
        logging.info(f"已添加新的快取項目: {question[:30]}...")
    
    def update_stats(self, question: str) -> None:
        """
        更新問題的訪問統計資料
        
        Args:
            question: 使用者問題
        """
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache_data:
            item = self.cache_data[cache_key]
            item['access_count'] = item.get('access_count', 0) + 1
            item['last_accessed'] = datetime.now().isoformat()
            self.cache_data[cache_key] = item
            
            # 每 10 次訪問保存一次快取（避免頻繁寫入）
            if item['access_count'] % 10 == 0:
                self._save_cache()
                
    def clear_cache(self) -> None:
        """清空快取"""
        self.cache_data = {}
        self.question_vectors = {}
        self._save_cache()
        logging.info("已清空所有快取")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        獲取快取統計資訊
        
        Returns:
            包含統計資訊的字典
        """
        stats = {
            'total_entries': len(self.cache_data),
            'cache_size_bytes': os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0,
            'vector_size_bytes': os.path.getsize(self.vector_file) if os.path.exists(self.vector_file) else 0,
            'source_type_counts': {},
            'oldest_entry': None,
            'newest_entry': None,
            'most_accessed': None
        }
        
        if not self.cache_data:
            return stats
            
        # 計算各種來源類型的數量
        for item in self.cache_data.values():
            source_type = item.get('source_type', 'unknown')
            stats['source_type_counts'][source_type] = stats['source_type_counts'].get(source_type, 0) + 1
            
        # 查找最舊和最新的快取
        sorted_by_time = sorted(
            [(k, v.get('timestamp', '')) for k, v in self.cache_data.items()],
            key=lambda x: x[1]
        )
        
        if sorted_by_time:
            oldest_key = sorted_by_time[0][0]
            newest_key = sorted_by_time[-1][0]
            stats['oldest_entry'] = {
                'question': self.cache_data[oldest_key].get('question', '')[:50],
                'timestamp': self.cache_data[oldest_key].get('timestamp', '')
            }
            stats['newest_entry'] = {
                'question': self.cache_data[newest_key].get('question', '')[:50],
                'timestamp': self.cache_data[newest_key].get('timestamp', '')
            }
            
        # 查找訪問最多的快取
        most_accessed_key = max(
            self.cache_data.items(),
            key=lambda x: x[1].get('access_count', 0) if x[1].get('access_count') else 0,
            default=(None, {})
        )[0]
        
        if most_accessed_key:
            stats['most_accessed'] = {
                'question': self.cache_data[most_accessed_key].get('question', '')[:50],
                'access_count': self.cache_data[most_accessed_key].get('access_count', 0)
            }
            
        return stats
