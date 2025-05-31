"""
RAG管理器
統一管理檢索增強生成(RAG)相關功能
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
from modules.rag.document_processor import DocumentProcessor
from modules.rag.vector_store import VectorStore
from modules.rag.retriever import Retriever

# 導入網路搜索服務
from modules.web_search.search_service import SearchService

#   ~看單元五~~
class RAGManager:
    """
    RAG管理器類，統一管理文檔處理、向量資料庫和檢索器
    """
    
    def __init__(self, knowledge_base_dir: Optional[str] = None):
        """
        初始化RAG管理器
        
        Args:
            knowledge_base_dir: 知識庫目錄路徑（可選）
        """
        # 設置知識庫目錄
        self.knowledge_base_dir = knowledge_base_dir or os.path.join(project_root, "data", "knowledge_base")
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # 設置向量資料庫文件路徑
        self.vector_db_path = os.path.join(project_root, "data", "vector_store", "vector_db")
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        
        # 初始化文檔處理器
        self.document_processor = DocumentProcessor(
            chunk_size=config.EMBEDDING_CHUNK_SIZE,
            chunk_overlap=config.EMBEDDING_CHUNK_OVERLAP
        )
        
        # 嘗試加載現有的向量資料庫，如果不存在則創建新的
        try:
            if os.path.exists(f"{self.vector_db_path}.index") and os.path.exists(f"{self.vector_db_path}.docs"):
                self.vector_store = VectorStore.load(self.vector_db_path)
                logging.info(f"已加載向量資料庫，包含 {len(self.vector_store.documents)} 個文檔")
            else:
                self.vector_store = VectorStore()
                logging.info("已創建新的向量資料庫")
        except Exception as e:
            logging.error(f"加載向量資料庫時出錯: {e}")
            self.vector_store = VectorStore()
            logging.info("已創建新的向量資料庫（由於加載錯誤）")
        
        # 初始化檢索器
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=3
        )
    
    def add_document(self, file_path: str) -> bool:
        """
        將單個文檔添加到知識庫
        
        Args:
            file_path: 文檔路徑
            
        Returns:
            是否成功添加文檔
        """
        try:
            # 檢查文件是否存在
            if not os.path.exists(file_path):
                logging.error(f"文件不存在: {file_path}")
                return False
                
            # 檢查文件類型
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.pdf', '.txt', '.docx', '.doc']:
                logging.error(f"不支持的文件類型: {file_ext}")
                return False
            
            # 准備目標文件路徑 - 將文件複製到知識庫目錄
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(self.knowledge_base_dir, file_name)
            
            # 如果源文件不在知識庫目錄，則複製到知識庫目錄
            if file_path != dest_path:
                import shutil
                shutil.copy2(file_path, dest_path)
                logging.info(f"文件已複製到知識庫目錄: {dest_path}")
            
            # 處理文檔
            logging.info(f"開始處理文檔: {file_path}")
            chunks = self.document_processor.process_file(file_path)
            logging.info(f"文檔分割為 {len(chunks)} 個區塊")
            
            # 將文檔添加到向量資料庫
            if chunks:
                self.vector_store.add_documents(chunks)
                
                # 保存向量資料庫
                self.vector_store.save(self.vector_db_path)
                
                logging.info(f"已成功添加文檔: {file_path}")
                return True
            else:
                logging.warning(f"文檔處理後沒有有效區塊: {file_path}")
                return False
        except Exception as e:
            logging.error(f"添加文檔時出錯: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def add_documents_from_directory(self, directory_path: Optional[str] = None) -> int:
        """
        將目錄中的所有文檔添加到知識庫
        
        Args:
            directory_path: 目錄路徑（可選，預設使用知識庫目錄）
            
        Returns:
            成功添加的文檔數量
        """
        directory_path = directory_path or self.knowledge_base_dir
        
        try:
            # 處理目錄中的所有文檔
            chunks = self.document_processor.process_directory(directory_path)
            
            # 將文檔添加到向量資料庫
            self.vector_store.add_documents(chunks)
            
            # 保存向量資料庫
            self.vector_store.save(self.vector_db_path)
            
            logging.info(f"已成功添加目錄中的文檔: {directory_path}")
            return len(chunks)
        except Exception as e:
            logging.error(f"添加目錄中的文檔時出錯: {e}")
            return 0
    
    def query(self, query: str) -> str:
        """
        查詢知識庫，獲取相關上下文
        
        Args:
            query: 查詢文本
            
        Returns:
            相關的上下文信息
        """
        try:
            # 檢查向量庫是否為空
            if not self.vector_store.documents:
                logging.warning("知識庫為空，無法查詢")
                return ""
            
            # 將查詢細分為和切分為多個上下文句子
            # 如果查詢述張述語超過 20 個字，嘗試提取关键句子
            import re
            sentences = re.split(r'[.!?！？。]', query)  # 依照標點符號切分句子
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # 去除空白區域和短區域
            
            # 如果有多個句子，將它們作為附加查詢
            contexts = []
            
            # 先對原始查詢進行查詢
            primary_context = self.retriever.get_context_for_query(query)
            if primary_context:
                contexts.append(primary_context)
            
            # 如果有多個句子，也對每個句子進行查詢
            if len(sentences) > 1:
                for sentence in sentences[:3]:  # 限制句子數量，避免過多查詢
                    if len(sentence) > 10:  # 只對較長的句子進行上下文查詢
                        sent_context = self.retriever.get_context_for_query(sentence)
                        if sent_context and sent_context not in contexts:
                            contexts.append(sent_context)
            
            # 將所有上下文合併
            if not contexts:
                return ""
            
            # 如果只有一個上下文，直接返回
            if len(contexts) == 1:
                return contexts[0]
            
            # 合併多個上下文
            combined_context = "以下是與您問題相關的各個資料段落：\n\n"
            for i, ctx in enumerate(contexts, 1):
                # 從上下文中去除標題部分，以免重複
                ctx_content = ctx.replace("以下是與您問題相關的資訊：\n\n", "")
                combined_context += f"\n---- 相關資料 {i} ----\n{ctx_content}\n"
            
            return combined_context
            
        except Exception as e:
            logging.error(f"查詢知識庫時出錯: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""
    
    def get_prompt_with_context(self, query: str, use_web_search: bool = False) -> str:
        """
        獲取帶有上下文的提示
        
        Args:
            query: 查詢文本
            use_web_search: 是否使用網路搜索
            
        Returns:
            帶有上下文的提示
        """
        # 獲取查詢的上下文信息
        context = self.query(query)
        
        # 如果啟用網路搜索且本地知識庫沒有足夠資訊，則使用網路搜索
        web_context = ""
        if use_web_search and (not context or len(context) < 200):
            try:
                # 使用 SerpAPI 進行網路搜索
                logging.info(f"從網路搜索相關資訊: {query}")
                search_service = SearchService()
                search_results = search_service.enrich_query_with_search(query)
                
                if search_results["success"]:
                    web_context = search_results["search_context"]
                    logging.info("成功獲取網路搜索結果")
                else:
                    logging.warning(f"網路搜索失敗: {search_results['search_context']}")
            except Exception as e:
                logging.error(f"網路搜索時出錯: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        # 組合本地知識庫和網路搜索的上下文
        combined_context = ""
        
        if context:
            combined_context += "從本地知識庫找到的資訊:\n" + context + "\n\n"
        
        if web_context:
            combined_context += web_context + "\n\n"
        
        # 如果沒有獲取到任何上下文，則直接返回查詢
        if not combined_context:
            return query
        
        # 格式化帶有上下文的提示
        prompt = f"{combined_context}根據以上資訊，請回答以下問題：\n\n{query}"
        
        return prompt
