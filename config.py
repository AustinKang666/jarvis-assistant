"""
JARVIS AI 助手設定檔案
"""
import os
from dotenv import load_dotenv
# 載入環境變數
load_dotenv()

# OpenAI API 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# RAG 設定
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
EMBEDDING_CHUNK_SIZE = int(os.getenv("EMBEDDING_CHUNK_SIZE", "1000"))
EMBEDDING_CHUNK_OVERLAP = int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "200"))

# 回應快取設定
RESPONSE_CACHE_FILE = os.getenv("RESPONSE_CACHE_FILE", "data/cache/response_cache.json")
RESPONSE_CACHE_MAX_SIZE = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "100"))
RESPONSE_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RESPONSE_CACHE_SIMILARITY_THRESHOLD", "0.85"))

# 網路搜尋設定
SERPAPI_KEY = os.getenv("SERPAPI_KEY","")
# ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "False").lower() in ('true', '1', 't')
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
# WEB_SEARCH_TOP_K = int(os.getenv("WEB_SEARCH_TOP_K", "3"))

# 安全過濾設定
SAFETY_FILTER_ENABLED = os.getenv("SAFETY_FILTER_ENABLED", "True").lower() in ('true', '1', 't')
SAFETY_FILTER_LEVEL = os.getenv("SAFETY_FILTER_LEVEL", "medium")  # low, medium, high

# 語音合成設定
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID= os.getenv("ELEVENLABS_VOICE_ID", "")  # ElevenLabs聲音ID

# 視覺模型設定
VISION_CONFIG = {
    "enabled": os.getenv("VISION_ENABLED", "True").lower() in ('true', '1', 't'),
    "model": os.getenv("VISION_MODEL", "gpt-4o-mini"),
    "max_image_size": 4 * 1024 * 1024,  # 4MB
    "supported_formats": ['.jpg', '.jpeg', '.png', '.webp', '.gif']
}

# 股票分析設定
STOCK_ANALYSIS_ENABLED = os.getenv("STOCK_ANALYSIS_ENABLED", "True").lower() in ('true', '1', 't')
STOCK_MODEL = os.getenv("STOCK_MODEL", "gpt-4o-mini")
