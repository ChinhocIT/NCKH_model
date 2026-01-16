# src/app/config.py
"""Configuration constants for GDA Application"""

from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class WebcamConfig:
    """Webcam settings"""
    width: int = 640
    height: int = 480
    fps: int = 10
    device_id: int = 0


@dataclass
class UIColors:
    """UI color constants (BGR format for OpenCV)"""
    # Status colors
    PROCESSING: Tuple[int, int, int] = (0, 165, 255)  # Orange
    WAITING_CLICK: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    MASK_READY: Tuple[int, int, int] = (0, 255, 0)  # Green
    DEFAULT: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Overlay colors
    MASK_OVERLAY: Tuple[int, int, int] = (0, 255, 0)  # Green
    BACKGROUND: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # Text colors
    ARCH_INFO: Tuple[int, int, int] = (200, 200, 200)  # Light gray
    QUERY_TEXT: Tuple[int, int, int] = (255, 200, 100)  # Light orange
    PREDICTION: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    
    # Recording
    REC_PULSE: Tuple[int, int, int] = (0, 0, 255)  # Red


@dataclass 
class UIStrings:
    """UI text strings (Vietnamese)"""
    # Status messages
    PROCESSING: str = "DANG XU LY... (vui long doi)"
    WAITING_CLICK: str = "CLICK vao vat the ban muon hoi"
    MASK_READY: str = "Giu C de noi cau hoi, hoac Enter de mo ta"
    IDLE: str = "Nhan SPACE de bat dau"
    
    # Recording
    RECORDING: str = "REC"
    HOLDING_C: str = "DANG GIU C"
    RELEASED_C: str = "DA THA C"
    
    # Window title
    WINDOW_TITLE: str = "GDA - Fixed Tensor Size"
    
    # Architecture info
    ARCH_TEXT: str = "Arch: Shared ViT -> [Seg Decoder + Adaptor] -> LLM"
    SAM_INFO: str = "SAM 2: Iterative Refinement | Debug: "
    SHORTCUTS: str = "Space: Chon | C+Voice: Hoi | Enter: Mo ta | S: Luu | D: Debug | Q: Thoat"


@dataclass
class KeyBindings:
    """Keyboard key codes"""
    QUIT: int = ord('q')
    SAVE: int = ord('s')
    DEBUG: int = ord('d')
    SPACE: int = ord(' ')
    ENTER: int = 13
    VOICE_KEY: str = 'c'


@dataclass
class InferenceConfig:
    """Inference settings"""
    queue_maxsize: int = 1
    voice_timeout: float = 4.0
    max_recording_time: float = 5.0
    min_audio_bytes: int = 500


@dataclass
class AppConfig:
    """Main application configuration"""
    # Default checkpoint paths
    default_seg_checkpoint: str = "D:/luu_tam/NCKH_model/checkpoints/seg_decoder_coco_stuff_best.pth"
    default_adaptor_checkpoint: str = None
    
    # Sub-configs
    webcam: WebcamConfig = field(default_factory=WebcamConfig)
    colors: UIColors = field(default_factory=UIColors)
    strings: UIStrings = field(default_factory=UIStrings)
    keys: KeyBindings = field(default_factory=KeyBindings)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # UI Layout
    status_bar_height: int = 100
    progress_bar_height: int = 10
    
    # Memory management
    frame_cleanup_interval: int = 30
    
    # Mask overlay
    mask_alpha: float = 0.3
