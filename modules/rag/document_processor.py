"""
文檔處理器模組
負責處理文檔的加載和分塊
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# 處理不同類型的文檔所需的庫
import docx
import PyPDF2
import re

# ~看單元五~~
class DocumentProcessor:
    """
    文檔處理器類，負責文檔加載和分塊
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文檔處理器
        
        Args:
            chunk_size: 文檔分塊大小
            chunk_overlap: 相鄰分塊的重疊字符數
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logging.info(f"初始化文檔處理器，塊大小: {chunk_size}，重疊大小: {chunk_overlap}")
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        處理單個文件，返回文檔分塊
        
        Args:
            file_path: 文件路徑
            
        Returns:
            文檔分塊列表，每個分塊是一個字典，包含文本和元數據
        """
        try:
            # 檢查文件是否存在
            if not os.path.exists(file_path):
                logging.error(f"文件不存在: {file_path}")
                return []
            
            # 獲取文件類型
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 加載文件內容
            if file_ext == '.txt':
                text = self._load_text_file(file_path)
            elif file_ext == '.pdf':
                text = self._load_pdf_file(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = self._load_docx_file(file_path)
            else:
                logging.error(f"不支持的文件類型: {file_ext}")
                return []
            
            # 如果文件內容為空，返回空列表
            if not text:
                logging.warning(f"文件內容為空: {file_path}")
                return []
            
            # 分割文本為塊
            chunks = self._split_text(text)
            
            # 創建文檔塊
            doc_chunks = []
            file_name = os.path.basename(file_path)
            
            for i, chunk in enumerate(chunks):
                # 去除空白行和多餘空白
                chunk = self._clean_text(chunk)
                
                # 如果塊不為空，則添加到結果中
                if chunk:
                    doc_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": file_name,
                            "chunk_id": i,
                            "file_path": file_path
                        }
                    })
            
            logging.info(f"文件 {file_path} 已處理為 {len(doc_chunks)} 個塊")
            return doc_chunks
            
        except Exception as e:
            logging.error(f"處理文件時出錯: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        處理目錄中的所有文件，返回所有文檔分塊
        
        Args:
            directory_path: 目錄路徑
            
        Returns:
            所有文檔分塊的列表
        """
        all_chunks = []
        
        try:
            # 檢查目錄是否存在
            if not os.path.isdir(directory_path):
                logging.error(f"目錄不存在: {directory_path}")
                return []
            
            # 獲取目錄中的所有文件
            files = []
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in ['.txt', '.pdf', '.docx', '.doc']:
                        files.append(file_path)
            
            # 處理每個文件
            for file_path in files:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            
            logging.info(f"目錄 {directory_path} 中的 {len(files)} 個文件已處理，共 {len(all_chunks)} 個塊")
            return all_chunks
            
        except Exception as e:
            logging.error(f"處理目錄時出錯: {e}")
            return []
    
    def _load_text_file(self, file_path: str) -> str:
        """
        加載文本文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            文件內容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except UnicodeDecodeError:
            # 如果 UTF-8 解碼失敗，嘗試其他編碼
            try:
                with open(file_path, 'r', encoding='big5') as f:
                    text = f.read()
                return text
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        text = f.read()
                    return text
                except:
                    logging.error(f"無法解碼文本文件: {file_path}")
                    return ""
        except Exception as e:
            logging.error(f"加載文本文件時出錯: {e}")
            return ""
    
    def _load_pdf_file(self, file_path: str) -> str:
        """
        加載PDF文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            文件內容
        """
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logging.error(f"加載PDF文件時出錯: {e}")
            return ""
    
    def _load_docx_file(self, file_path: str) -> str:
        """
        加載DOCX文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            文件內容
        """
        try:
            doc = docx.Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logging.error(f"加載DOCX文件時出錯: {e}")
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """
        將文本分割為塊
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本塊列表
        """
        # 如果文本長度小於塊大小，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        # 處理多餘換行和空格
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{3,}', ' ', text)
        
        # 優先按照段落分割
        paragraphs = text.split('\n\n')
        
        # 初始化結果
        chunks = []
        current_chunk = ""
        
        # 遍歷段落
        for para in paragraphs:
            # 如果段落加入當前塊後不超過最大長度，則添加到當前塊
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 如果當前段落本身就超過塊大小，需要進一步拆分
                if len(para) > self.chunk_size:
                    # 先保存當前塊
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # 按照句子拆分長段落
                    sentences = re.split(r'(?<=[。！？.!?])', para)
                    
                    # 組合句子成塊
                    temp_chunk = ""
                    for sentence in sentences:
                        if not sentence:
                            continue
                        
                        if len(temp_chunk) + len(sentence) <= self.chunk_size:
                            temp_chunk += sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            
                            # 如果句子本身超過塊大小，需要按照字符分割
                            if len(sentence) > self.chunk_size:
                                sentence_chunks = [sentence[i:i+self.chunk_size] for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap)]
                                chunks.extend(sentence_chunks)
                                temp_chunk = ""
                            else:
                                temp_chunk = sentence
                    
                    # 保存最後一個臨時塊
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    # 當前塊已經達到最大長度，保存並開始新塊
                    chunks.append(current_chunk)
                    current_chunk = para
        
        # 添加最後一個塊
        if current_chunk:
            chunks.append(current_chunk)
        
        # 處理塊的重疊部分
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i < len(chunks) - 1 and len(chunks[i]) + self.chunk_overlap <= self.chunk_size:
                    # 添加後一個塊的開頭部分到當前塊末尾
                    next_chunk_start = chunks[i + 1][:min(self.chunk_overlap, len(chunks[i + 1]))]
                    overlapped_chunks.append(chunks[i] + "\n" + next_chunk_start)
                else:
                    overlapped_chunks.append(chunks[i])
            
            return overlapped_chunks
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本，移除多餘空白和空行
        
        Args:
            text: 要清理的文本
            
        Returns:
            清理後的文本
        """
        # 處理多餘換行和空格
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{3,}', ' ', text)
        return text.strip()
