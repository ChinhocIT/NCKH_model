import cv2
from pynput import keyboard as pynput_keyboard
import time

class ClickHandler:
    def __init__(self):
        self.clicked_point = None
        self.waiting_for_click = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.waiting_for_click:
            self.clicked_point = (x, y)
            print(f"✓ Đã chọn điểm: ({x}, {y})")

# ============================================================================
# KEYBOARD MONITOR
# ============================================================================

class KeyboardMonitor:
    def __init__(self):
        self.c_pressed = False
        self.c_last_press_time = 0
        self.listener = None
    
    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char == 'c':
                now = time.time()
                if now - self.c_last_press_time > 1.0:
                    self.c_pressed = True
                    self.c_last_press_time = now
        except:
            pass
    
    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char == 'c':
                self.c_pressed = False
        except:
            pass
    
    def start(self):
        self.listener = pynput_keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def stop(self):
        if self.listener:
            self.listener.stop()