import numpy as np

# Import constants từ src.constants
from ..constants import COCO_STUFF_CLASSES

class PromptConstructor:
    """Tạo prompt CHẤT LƯỢNG CAO với spatial + semantic context"""
    
    def __init__(self, class_names=COCO_STUFF_CLASSES):
        self.class_names = ['nền'] + class_names
        
        self.question_patterns = {
        'what_is': [
            'đây là gì', 'đó là gì', 'cái này là gì', 'vật này là gì',
            'what is', 'là gì', 'cái gì', 'thứ gì'
        ],
        'color': [
            'màu gì', 'màu sắc', 'màu nào', 'color', 'mau',
            'màu của', 'có màu'
        ],
        'describe': [
            'mô tả', 'describe', 'chi tiết', 'mo ta',
            'miêu tả', 'nói về', 'giới thiệu'
        ],
        'material': [
            'làm bằng gì', 'chất liệu', 'material', 'chat lieu',
            'vật liệu', 'bằng gì'
        ],
        'shape': [
            'hình dạng', 'hình gì', 'shape', 'hinh dang',
            'hình thù', 'dạng'
        ],
        'size': [
            'kích thước', 'to hay nhỏ', 'size', 'kich thuoc',
            'bao lớn', 'cỡ nào', 'độ lớn'
        ],
        'location': [
            'ở đâu', 'vị trí', 'where', 'vi tri',
            'nằm ở', 'chỗ nào', 'đâu'
        ],
        'count': [
            'bao nhiêu', 'how many', 'số lượng', 'so luong',
            'mấy', 'có mấy', 'bao nhiều'
        ]
    }
    
    def detect_question_type(self, user_query):
        if not user_query:
            return 'general_describe'
        
        query_lower = user_query.lower()
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return q_type
        return 'general_describe'
    
    def _get_spatial_context(self, mask):
        """Phân tích vị trí và kích thước chi tiết"""
        h, w = mask.shape
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) == 0:
            return "ở vị trí không xác định", 0.0
        
        # Spatial info
        center_x = x_coords.mean() / w
        center_y = y_coords.mean() / h
        area_ratio = len(y_coords) / (h * w)
        
        # Vị trí
        if center_y < 0.33:
            v_pos = "phía trên"
        elif center_y < 0.67:
            v_pos = "ở giữa"
        else:
            v_pos = "phía dưới"
        
        if center_x < 0.33:
            h_pos = "bên trái"
        elif center_x < 0.67:
            h_pos = "ở trung tâm"
        else:
            h_pos = "bên phải"
        
        # Kích thước
        if area_ratio > 0.4:
            size_desc = "chiếm phần lớn khung hình"
        elif area_ratio > 0.2:
            size_desc = "có kích thước trung bình"
        elif area_ratio > 0.05:
            size_desc = "có kích thước nhỏ"
        else:
            size_desc = "rất nhỏ"
        
        location = f"{v_pos} {h_pos} khung hình, {size_desc}"
        
        return location, area_ratio
    
    def construct_prompt(self, mask, predicted_class=None, user_query=None):
        """
        ULTRA FOCUSED - TẬP TRUNG VÀO VÙNG MÀU XANH
        """
        location, area_ratio = self._get_spatial_context(mask)
        q_type = self.detect_question_type(user_query)
        
        # ===== BASE PROMPT - NHẤN MẠNH VÙNG =====
        base_instruction = """QUAN TRỌNG: Trong ảnh có một vùng được tô màu xanh lá cây và viền đỏ. 
    Đó là vật thể tôi muốn hỏi về. HÃY CHỈ MÔ TẢ VẬT THỂ TRONG VÙNG ĐÓ, không mô tả phần còn lại của ảnh.

    """
        
        if q_type == 'what_is' or not user_query:
            # Mô tả tổng quát
            prompt = base_instruction + f"""Vật thể trong vùng màu xanh nằm {location}.
    {f"Có thể là: {predicted_class}." if predicted_class else ""}

    Hãy mô tả VẬT THỂ TRONG VÙNG MÀU XANH:
    - Đó là gì?
    - Màu sắc chính?
    - Hình dạng/đặc điểm nổi bật?

    Trả lời ngắn gọn 2-3 câu:"""
            
        else:
            # Câu hỏi cụ thể
            prompt = base_instruction + f"""Câu hỏi: "{user_query}"

    Vật thể trong vùng màu xanh nằm {location}.

    HÃY TRẢ LỜI VỀ VẬT THỂ TRONG VÙNG MÀU XANH (2-3 câu):"""
        
        return prompt