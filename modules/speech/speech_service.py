"""
語音服務模組
提供語音識別和語音合成功能
"""
import os
import sys
import io
import json
import logging
from pathlib import Path
import tempfile
import wave
import numpy as np
import requests
from typing import Tuple, Optional, Union, Dict, Any

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import config
from openai import OpenAI

class SpeechService:
    """
    語音服務類
    提供語音識別(STT)和語音合成(TTS)功能
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化語音服務
        
        Args:
            api_key: OpenAI API 密鑰，如果不提供則從配置檔案獲取
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.api_key)
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
        
        # 檢查配置
        if not self.api_key:
            logging.warning("未設置 OpenAI API 密鑰，語音識別功能將不可用")
            
        # 確保臨時目錄存在
        self.temp_dir = Path(project_root) / "data" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def speech_to_text(self, audio_data: Union[bytes, str], 
                      mime_type: str = "audio/wav") -> Dict[str, Any]:
        """
        將語音轉換為文本
        
        Args:
            audio_data: 音頻數據（字節）或音頻文件路徑
            mime_type: 音頻MIME類型
            
        Returns:
            包含轉錄文本的字典，格式: {'text': '轉錄文本', 'status': '成功/失敗'}
        """
        if not self.api_key:
            return {"text": "", "status": "failed", "error": "未設置 OpenAI API 密鑰"}
        
        try:
            # 如果輸入是文件路徑，讀取文件
            if isinstance(audio_data, str):
                with open(audio_data, "rb") as audio_file:
                    audio_bytes = audio_file.read()
            else:
                audio_bytes = audio_data
            
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # 使用OpenAI的Whisper模型轉錄
            with open(temp_file_path, "rb") as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="zh"  # 使用繁體中文
                )
            
            # 清理臨時文件
            os.unlink(temp_file_path)
            
            return {"text": response.text, "status": "success"}
            
        except Exception as e:
            logging.error(f"語音轉文本失敗: {str(e)}")
            return {"text": "", "status": "failed", "error": str(e)}
    
    def text_to_speech(self, text: str, voice_id: Optional[str] = None, 
                       use_elevenlabs: bool = False) -> Tuple[bytes, str]:
        """
        將文本轉換為語音
        
        Args:
            text: 要轉換的文本
            voice_id: 語音ID（如果使用ElevenLabs）
            use_elevenlabs: 是否使用ElevenLabs進行語音合成
            
        Returns:
            語音數據（字節）和MIME類型
        """
        # 為簡化目的，將只使用 OpenAI TTS
        return self._openai_tts(text)
    
    def _openai_tts(self, text: str, voice: str = "alloy") -> Tuple[bytes, str]:
        """
        使用OpenAI的TTS API生成語音
        
        Args:
            text: 要轉換的文本
            voice: 語音類型 (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            語音數據（字節）和MIME類型
        """
        if not self.api_key:
            raise ValueError("未設置 OpenAI API 密鑰")
        
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            # 獲取音頻數據
            audio_data = response.content
            
            return audio_data, "audio/mp3"
            
        except Exception as e:
            logging.error(f"OpenAI 文本轉語音失敗: {str(e)}")
            raise
    
    def _elevenlabs_tts(self, text: str, voice_id: str = None) -> Tuple[bytes, str]:
        """
        使用ElevenLabs的API生成語音
        
        Args:
            text: 要轉換的文本
            voice_id: ElevenLabs語音ID
            
        Returns:
            語音數據（字節）和MIME類型
        """
        if not self.elevenlabs_api_key:
            raise ValueError("未設置 ElevenLabs API 密鑰")
        
        # 如果未提供語音ID，使用默認值
        if not voice_id:
            if not self.elevenlabs_voice_id:
                raise ValueError("未設置 ElevenLabs 語音ID")
            voice_id = self.elevenlabs_voice_id
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            
            return response.content, "audio/mpeg"
            
        except Exception as e:
            logging.error(f"ElevenLabs 文本轉語音失敗: {str(e)}")
            # 失敗時回退到OpenAI TTS
            logging.info("回退到 OpenAI 文本轉語音")
            return self._openai_tts(text)

    def detect_silence(self, audio_data: bytes, 
                      threshold: int = 3500,
                      sample_width: int = 2) -> bool:
        """
        檢測音頻是否包含靜音
        
        Args:
            audio_data: 音頻數據（字節）
            threshold: 音量閾值
            sample_width: 樣本寬度（字節）
            
        Returns:
            是否為靜音
        """
        try:
            import audioop
            
            # 計算音頻能量
            energy = audioop.rms(audio_data, sample_width)
            
            # 如果能量低於閾值，認為是靜音
            return energy < threshold
            
        except Exception as e:
            logging.error(f"檢測靜音失敗: {str(e)}")
            return False
            
    def get_audio_duration(self, audio_data: bytes) -> float:
        """
        獲取音頻持續時間
        
        Args:
            audio_data: 音頻數據（字節）
            
        Returns:
            音頻持續時間（秒）
        """
        try:
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 打開WAV文件
            with wave.open(temp_file_path, 'rb') as wav_file:
                # 獲取幀數和幀率
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                
                # 計算持續時間
                duration = frames / float(rate)
            
            # 清理臨時文件
            os.unlink(temp_file_path)
            
            return duration
            
        except Exception as e:
            logging.error(f"獲取音頻持續時間失敗: {str(e)}")
            return 0.0
            
    def save_audio(self, audio_data: bytes, file_path: str) -> bool:
        """
        保存音頻數據到文件
        
        Args:
            audio_data: 音頻數據（字節）
            file_path: 文件保存路徑
            
        Returns:
            是否保存成功
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            return True
        except Exception as e:
            logging.error(f"保存音頻失敗: {str(e)}")
            return False

# 測試函數
def test_speech_service():
    """
    測試語音服務功能
    """
    service = SpeechService()
    
    # 測試文本轉語音
    try:
        text = "你好，我是 JARVIS 語音助手。很高興為您服務。"
        audio_data, mime_type = service._openai_tts(text)
        print(f"生成了 {len(audio_data)} 字節的音頻，MIME類型: {mime_type}")
        
        # 保存音頻供測試
        output_path = os.path.join(project_root, 'data', 'temp', 'test_output.mp3')
        service.save_audio(audio_data, output_path)
        print(f"保存測試音頻到: {output_path}")
    except Exception as e:
        print(f"文本轉語音測試失敗: {str(e)}")

if __name__ == "__main__":
    # 執行測試
    test_speech_service()
