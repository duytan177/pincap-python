import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class APIKeyManager:
    """
    Quản lý nhiều API keys và tự động rotate khi gặp lỗi 429 (rate limit).
    Hỗ trợ nhiều API keys từ environment variables.
    """
    
    def __init__(self, env_key_prefix: str = "GEMINI_API_KEY", max_keys: int = 5):
        """
        Khởi tạo APIKeyManager.
        
        Args:
            env_key_prefix: Prefix của environment variable (mặc định: GEMINI_API_KEY)
                           Sẽ tìm GEMINI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
            max_keys: Số lượng keys tối đa để tìm (mặc định: 5)
        """
        self.keys: List[str] = []
        self.current_index = 0
        self.failed_keys = set()  # Lưu các keys đã bị 429 để tạm thời bỏ qua
        
        # Tìm tất cả các keys từ environment
        # Ưu tiên GEMINI_API_KEY (key chính)
        main_key = os.getenv(env_key_prefix)
        if main_key:
            self.keys.append(main_key)
        
        # Tìm các keys phụ: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        for i in range(1, max_keys + 1):
            key = os.getenv(f"{env_key_prefix}_{i}")
            if key and key not in self.keys:
                self.keys.append(key)
        
        if not self.keys:
            raise ValueError(f"Không tìm thấy API key nào với prefix '{env_key_prefix}'. "
                           f"Vui lòng đặt {env_key_prefix} hoặc {env_key_prefix}_1, {env_key_prefix}_2, ...")
        
        print(f"✅ Đã load {len(self.keys)} API key(s) cho {env_key_prefix}", flush=True)
    
    def get_current_key(self) -> str:
        """Lấy API key hiện tại"""
        if not self.keys:
            raise ValueError("Không có API key nào available")
        
        # Lọc bỏ các keys đã bị 429
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            # Nếu tất cả keys đều bị 429, reset và thử lại từ đầu
            print("⚠️ Tất cả keys đều bị rate limit, reset và thử lại...", flush=True)
            self.failed_keys.clear()
            available_keys = self.keys
        
        # Lấy key hiện tại từ available keys
        if self.current_index >= len(available_keys):
            self.current_index = 0
        
        return available_keys[self.current_index]
    
    def mark_key_failed(self, key: Optional[str] = None):
        """
        Đánh dấu một key bị lỗi 429 (rate limit).
        
        Args:
            key: Key cụ thể cần đánh dấu. Nếu None, sẽ đánh dấu key hiện tại.
        """
        if key is None:
            key = self.get_current_key()
        
        if key in self.keys:
            self.failed_keys.add(key)
            print(f"⚠️ Đánh dấu key bị rate limit (429). Đang chuyển sang key khác...", flush=True)
    
    def rotate_to_next_key(self) -> str:
        """
        Chuyển sang key tiếp theo và trả về key mới.
        
        Returns:
            API key mới
        """
        # Lấy danh sách available keys (bỏ qua failed keys)
        available_keys = [k for k in self.keys if k not in self.failed_keys]
        
        if not available_keys:
            # Nếu tất cả keys đều bị failed, reset và dùng lại
            print("🔄 Tất cả keys đều bị rate limit, reset và thử lại...", flush=True)
            self.failed_keys.clear()
            available_keys = self.keys
        
        # Tìm index của key hiện tại trong available keys
        current_key = self.get_current_key()
        try:
            current_idx_in_available = available_keys.index(current_key)
            # Chuyển sang key tiếp theo trong available keys
            next_idx = (current_idx_in_available + 1) % len(available_keys)
            self.current_index = self.keys.index(available_keys[next_idx])
        except (ValueError, IndexError):
            # Nếu không tìm thấy, chuyển sang key đầu tiên trong available
            self.current_index = self.keys.index(available_keys[0])
        
        new_key = self.get_current_key()
        print(f"🔄 Đã chuyển sang API key mới (index: {self.current_index}/{len(self.keys)})", flush=True)
        
        return new_key
    
    def reset_failed_keys(self):
        """Reset tất cả failed keys (sau một khoảng thời gian)"""
        if self.failed_keys:
            print(f"🔄 Reset {len(self.failed_keys)} failed key(s)", flush=True)
            self.failed_keys.clear()
    
    def get_available_count(self) -> int:
        """Trả về số lượng keys còn available (chưa bị 429)"""
        return len([k for k in self.keys if k not in self.failed_keys])


# Singleton instance cho Gemini API keys
_gemini_key_manager: Optional[APIKeyManager] = None


def get_gemini_key_manager() -> APIKeyManager:
    """Lấy singleton instance của Gemini APIKeyManager"""
    global _gemini_key_manager
    if _gemini_key_manager is None:
        _gemini_key_manager = APIKeyManager(env_key_prefix="GEMINI_API_KEY", max_keys=5)
    return _gemini_key_manager

