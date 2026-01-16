import torch
import cv2
import numpy as np
from typing import Tuple
from transformers import SamModel, SamProcessor

class SAM2Segmenter:
    """
    SAM 2 với iterative refinement cho mask chất lượng cao
    """
    
    def __init__(self, model_name="facebook/sam-vit-huge", device="cuda"):
        self.device = device
        print("🎯 Đang tải SAM 2 model...")
        
        try:
            self.sam_model = SamModel.from_pretrained(model_name).to(device)
            self.sam_processor = SamProcessor.from_pretrained(model_name)
            self.sam_model.eval()
            print(f"✅ SAM 2 ready: {model_name}")
        except Exception as e:
            print(f"⚠️ SAM 2 không khả dụng: {e}")
            print("   Fallback: GrabCut")
            self.sam_model = None
            self.sam_processor = None
    
    @torch.inference_mode()
    def segment_from_point(self, image_rgb: np.ndarray, 
                        point: Tuple[int, int],
                        use_iterative=False) -> np.ndarray:
        """
        Phân đoạn từ điểm click với iterative refinement
        """
        if self.sam_model is None:
            return self._grabcut_segment(image_rgb, point)
        
        try:
            if use_iterative:
                return self._segment_iterative(image_rgb, point)
            else:
                return self._segment_single(image_rgb, point)
        except Exception as e:
            print(f"⚠️ SAM 2 error: {e}, fallback to GrabCut")
            return self._grabcut_segment(image_rgb, point)
    
    def _segment_single(self, image_rgb, point):
        """Single-point segmentation"""
        inputs = self.sam_processor(
            image_rgb,
            input_points=[[[point[0], point[1]]]],
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        scores = outputs.iou_scores.cpu()[0, 0]
        best_mask_idx = scores.argmax().item()
        mask = masks[0, best_mask_idx].numpy()
        
        del outputs, masks, scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return mask.astype(np.uint8)
    
    def _segment_iterative(self, image_rgb, point, max_iterations=2):
        """Iterative refinement"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Iteration 1: Single point
        mask = self._segment_single(image_rgb, point)
        
        for iteration in range(max_iterations - 1):
            # Tìm boundary points để refine
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                break
            
            # Lấy contour lớn nhất
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 5:
                break
            
            # Sample 3 điểm trên boundary
            indices = np.linspace(0, len(largest_contour)-1, 3, dtype=int)
            boundary_points = [
                [int(largest_contour[idx][0][0]), int(largest_contour[idx][0][1])]
                for idx in indices
            ]
            
            # Multi-point prompting
            all_points = [[point[0], point[1]]] + boundary_points
            
            inputs = self.sam_processor(
                image_rgb,
                input_points=[all_points],
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.cpu()[0, 0]
            best_mask_idx = scores.argmax().item()
            mask_refined = masks[0, best_mask_idx].numpy().astype(np.uint8)
            
            # Combine masks (union)
            mask = np.logical_or(mask, mask_refined).astype(np.uint8)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return mask
    
    def _grabcut_segment(self, image_rgb, point):
        """Fallback: GrabCut segmentation"""
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y = point
        rect_size = min(w, h) // 4
        x1 = max(0, x - rect_size // 2)
        y1 = max(0, y - rect_size // 2)
        x2 = min(w, x + rect_size // 2)
        y2 = min(h, y + rect_size // 2)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except:
            cv2.circle(mask, (x, y), rect_size // 2, 1, -1)
        
        return mask