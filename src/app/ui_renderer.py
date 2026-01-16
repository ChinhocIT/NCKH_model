# src/app/ui_renderer.py
"""UI rendering for GDA Application"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple, List
from .config import AppConfig


class UIRenderer:
    """
    Handles all UI rendering to OpenCV frame.
    Separated from logic for cleaner code.
    """
    
    def __init__(self, config: AppConfig):
        """
        Args:
            config: Application configuration
        """
        self.config = config
        self.colors = config.colors
        self.strings = config.strings
    
    def draw_mask_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Draw mask overlay on frame.
        
        Args:
            frame: BGR frame
            mask: Binary mask
            
        Returns:
            Frame with mask overlay
        """
        if mask is None:
            return frame
            
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = self.colors.MASK_OVERLAY
        
        display = cv2.addWeighted(
            frame, 1 - self.config.mask_alpha, 
            overlay, self.config.mask_alpha, 
            0
        )
        
        # Draw contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, self.colors.MASK_OVERLAY, 2)
        
        return display
    
    def draw_status_bar(self, frame: np.ndarray, 
                        is_processing: bool,
                        progress: int,
                        waiting_for_click: bool,
                        has_mask: bool,
                        debug: bool) -> np.ndarray:
        """
        Draw the top status bar.
        
        Args:
            frame: BGR frame
            is_processing: Whether inference is running
            progress: Progress percentage
            waiting_for_click: Whether waiting for user click
            has_mask: Whether a mask is active
            debug: Debug mode flag
            
        Returns:
            Frame with status bar
        """
        # Determine status text and color
        if is_processing:
            status_text = f"{self.strings.PROCESSING} {progress}%"
            color = self.colors.PROCESSING
        elif waiting_for_click:
            status_text = self.strings.WAITING_CLICK
            color = self.colors.WAITING_CLICK
        elif has_mask:
            status_text = self.strings.MASK_READY
            color = self.colors.MASK_READY
        else:
            status_text = self.strings.IDLE
            color = self.colors.DEFAULT
        
        # Draw background
        cv2.rectangle(frame, (5, 5), (630, 100), self.colors.BACKGROUND, -1)
        cv2.rectangle(frame, (5, 5), (630, 100), color, 2)
        
        # Draw status text
        cv2.putText(
            frame, status_text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Draw progress bar if processing
        if is_processing and progress > 0:
            self._draw_progress_bar(frame, progress)
        
        # Draw architecture info
        cv2.putText(
            frame, self.strings.ARCH_TEXT, (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.ARCH_INFO, 1
        )
        
        cv2.putText(
            frame, 
            self.strings.SAM_INFO + ("ON" if debug else "OFF"),
            (15, 82), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.ARCH_INFO, 1
        )
        
        cv2.putText(
            frame, self.strings.SHORTCUTS, (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.colors.DEFAULT, 1
        )
        
        return frame
    
    def _draw_progress_bar(self, frame: np.ndarray, progress: int):
        """Draw progress bar"""
        bar_width = 610
        bar_height = self.config.progress_bar_height
        bar_x = 10
        bar_y = 40
        
        # Background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50), -1
        )
        
        # Fill
        fill_width = int(bar_width * progress / 100)
        if fill_width > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + fill_width, bar_y + bar_height),
                self.colors.MASK_READY, -1
            )
    
    def draw_result(self, frame: np.ndarray, result: Dict, 
                    debug: bool = False) -> np.ndarray:
        """
        Draw inference result on frame.
        
        Args:
            frame: BGR frame
            result: Result dictionary
            debug: Debug mode flag
            
        Returns:
            Frame with result overlay
        """
        if not result or result.get('error'):
            return frame
        
        info_y = 110
        
        # Debug info - ViT features shape
        if debug and result.get('vit_features_shape'):
            vit_shape = result['vit_features_shape']
            vision_tokens_shape = result['vision_tokens_shape']
            
            arch_info = f"ViT: {vit_shape} -> Tokens: {vision_tokens_shape}"
            cv2.rectangle(frame, (5, info_y), (450, info_y + 25), (30, 30, 60), -1)
            cv2.putText(
                frame, arch_info, (10, info_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1
            )
            info_y += 30
        
        # Predicted class
        if result.get('predicted_class'):
            conf = result.get('confidence', 0)
            pred_text = f"Phat hien: {result['predicted_class']} ({conf:.0%})"
            cv2.rectangle(frame, (5, info_y), (450, info_y + 25), (50, 50, 50), -1)
            cv2.putText(
                frame, pred_text, (10, info_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors.PREDICTION, 1
            )
            info_y += 30
        
        # Query
        query_text = f"Cau hoi: {result['query']}"
        cv2.rectangle(frame, (5, info_y), (630, info_y + 25), (50, 50, 50), -1)
        cv2.putText(
            frame, query_text, (10, info_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.QUERY_TEXT, 1
        )
        info_y += 30
        
        # Description (wrapped)
        desc = result.get('description', '')
        lines = self._wrap_text(desc, max_chars=75, max_lines=4)
        
        box_height = 30 + len(lines) * 20
        cv2.rectangle(frame, (5, info_y), (635, info_y + box_height), (0, 50, 0), -1)
        cv2.rectangle(frame, (5, info_y), (635, info_y + box_height), self.colors.MASK_READY, 1)
        
        cv2.putText(
            frame, "Tra loi:", (10, info_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.PREDICTION, 1
        )
        
        y_offset = info_y + 38
        for line in lines:
            cv2.putText(
                frame, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.colors.DEFAULT, 1
            )
            y_offset += 20
        
        return frame
    
    def draw_recording_indicator(self, frame: np.ndarray, 
                                  elapsed: float,
                                  c_pressed: bool) -> np.ndarray:
        """
        Draw recording indicator during voice capture.
        
        Args:
            frame: BGR frame
            elapsed: Elapsed time in seconds
            c_pressed: Whether C key is pressed
            
        Returns:
            Frame with recording overlay
        """
        # Pulsing red circle
        pulse = int(abs(np.sin(elapsed * 4) * 255))
        cv2.circle(frame, (30, 30), 15, (0, 0, pulse), -1)
        
        # Time display
        cv2.putText(
            frame, f"REC {elapsed:.1f}s", (55, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors.REC_PULSE, 2
        )
        
        # C key status
        c_status = self.strings.HOLDING_C if c_pressed else self.strings.RELEASED_C
        c_color = self.colors.MASK_READY if c_pressed else self.colors.REC_PULSE
        cv2.putText(
            frame, c_status, (150, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_color, 2
        )
        
        return frame
    
    def _wrap_text(self, text: str, max_chars: int = 75, 
                   max_lines: int = 4) -> List[str]:
        """Wrap text into lines"""
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines[:max_lines]
