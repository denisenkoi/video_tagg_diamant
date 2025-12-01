#!/usr/bin/env python3
"""
Hunyuan-MT-7B Translation Client
World's best open-source translation model (WMT25 winner)
"""

import requests
import json
import time
from typing import Optional, Dict, Any

class HunyuanTranslator:
    def __init__(self, api_base: str = "http://127.0.0.1:11435"):
        self.api_base = api_base
        self.model_name = "tencent/Hunyuan-MT-7B"
        self.session = requests.Session()
        
        # Optimal inference parameters based on Tencent recommendations
        self.default_params = {
            "temperature": 0.7,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": 2048
        }
        
        print(f"✓ HunyuanTranslator initialized")
        print(f"  API: {self.api_base}")
        print(f"  Model: {self.model_name}")
    
    def test_connection(self) -> bool:
        """Test connection to Hunyuan-MT API"""
        try:
            response = self.session.get(f"{self.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✓ Hunyuan-MT API connection successful")
                return True
            else:
                print(f"✗ Hunyuan-MT API returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Hunyuan-MT API connection failed: {type(e).__name__}")
            return False
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "Russian") -> Optional[str]:
        """
        Translate text using Hunyuan-MT-7B
        
        Args:
            text: Text to translate
            source_lang: Source language (auto-detect if "auto")
            target_lang: Target language
            
        Returns:
            Translated text or None if failed
        """
        if not text.strip():
            return None
        
        # Create translation prompt
        if source_lang == "auto":
            prompt = f"Please translate the following text to {target_lang}:\n\n{text}"
        else:
            prompt = f"Please translate the following {source_lang} text to {target_lang}:\n\n{text}"
        
        # Prepare request
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self.default_params
        }
        
        try:
            response = self.session.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if content:
                        # Clean up the response - remove any explanatory text
                        cleaned = content.strip()
                        return cleaned
                    
            print(f"    ❌ Hunyuan-MT translation failed: HTTP {response.status_code}")
            return None
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"    ❌ Hunyuan-MT connection error: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"    ❌ Hunyuan-MT translation error: {type(e).__name__}: {e}")
            return None
    
    def translate_to_russian(self, text: str, source_lang: str = "auto") -> Optional[str]:
        """Convenience method to translate to Russian"""
        return self.translate(text, source_lang, "Russian")
    
    def translate_chinese_to_russian(self, text: str) -> Optional[str]:
        """Translate Chinese text to Russian"""
        return self.translate(text, "Chinese", "Russian")
    
    def translate_english_to_russian(self, text: str) -> Optional[str]:
        """Translate English text to Russian"""
        return self.translate(text, "English", "Russian")


def test_hunyuan_translator():
    """Test function for Hunyuan translator"""
    print("🔬 Testing Hunyuan-MT-7B Translation...")
    
    translator = HunyuanTranslator()
    
    # Test connection
    if not translator.test_connection():
        print("❌ Cannot connect to Hunyuan-MT API")
        return
    
    # Test translations
    test_cases = [
        ("Hello world!", "English"),
        ("你好世界！", "Chinese"), 
        ("Bonjour le monde!", "French"),
        ("This is a test of the world's best translation model.", "English")
    ]
    
    for text, lang in test_cases:
        print(f"\n📝 Testing {lang} → Russian:")
        print(f"   Input: {text}")
        
        result = translator.translate_to_russian(text, lang)
        if result:
            print(f"   Output: {result}")
        else:
            print(f"   ❌ Translation failed")
    
    print("\n🎉 Hunyuan-MT-7B testing completed!")


if __name__ == "__main__":
    test_hunyuan_translator()