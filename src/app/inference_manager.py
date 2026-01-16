# src/app/inference_manager.py
"""Inference queue manager for GDA Application"""

import torch
import threading
import queue
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Container for inference results"""
    processing: bool = False
    result: Optional[Dict] = None
    progress: int = 0


class InferenceManager:
    """
    Manages background inference worker and result queue.
    Thread-safe design with Lock for state access.
    """
    
    def __init__(self, gda_system, maxsize: int = 1):
        """
        Args:
            gda_system: GlobalDescriptionAcquisition instance
            maxsize: Maximum queue size
        """
        self.gda = gda_system
        self.queue = queue.Queue(maxsize=maxsize)
        self.result = InferenceResult()
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start the inference worker thread"""
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop, 
            daemon=True,
            name="InferenceWorker"
        )
        self.worker_thread.start()
    
    def stop(self):
        """Stop the worker thread gracefully"""
        self._running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
    
    def submit(self, frame_rgb, mask, user_query: Optional[str] = None) -> bool:
        """
        Submit a task to the inference queue.
        
        Returns:
            True if submitted successfully, False if queue is full or busy
        """
        with self.lock:
            if self.result.processing:
                return False
        
        try:
            self.queue.put_nowait((frame_rgb.copy(), mask.copy(), user_query))
            return True
        except queue.Full:
            return False
    
    def get_status(self) -> tuple:
        """
        Get current processing status.
        
        Returns:
            (is_processing, progress, result)
        """
        with self.lock:
            return (
                self.result.processing,
                self.result.progress,
                self.result.result
            )
    
    def consume_result(self) -> Optional[Dict]:
        """
        Get and clear the result if available.
        
        Returns:
            Result dict or None
        """
        with self.lock:
            if not self.result.processing and self.result.result is not None:
                result = self.result.result
                self.result.result = None
                return result
        return None
    
    def is_processing(self) -> bool:
        """Check if currently processing"""
        with self.lock:
            return self.result.processing
    
    def _worker_loop(self):
        """Main worker loop - runs in background thread"""
        while self._running:
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if task is None:
                break
            
            frame_rgb, mask, user_query = task
            
            # Initialize processing state
            with self.lock:
                self.result.processing = True
                self.result.result = None
                self.result.progress = 0
            
            try:
                query_text = user_query if user_query else 'Mô tả tự động'
                print(f"\n🔄 Worker: {query_text}")
                
                # Update progress
                with self.lock:
                    self.result.progress = 10
                
                # Run inference
                result = self.gda.process_region(frame_rgb, mask, user_query)
                
                with self.lock:
                    self.result.progress = 95
                
                print("✅ Worker hoàn thành")
                
            except Exception as e:
                print(f"❌ Worker lỗi: {e}")
                if self.gda.debug:
                    import traceback
                    traceback.print_exc()
                
                result = {
                    'description': f"Lỗi: {str(e)}",
                    'error': True,
                    'predicted_class': None,
                    'confidence': 0.0,
                    'query': user_query if user_query else "Lỗi"
                }
            
            finally:
                with self.lock:
                    self.result.processing = False
                    self.result.result = result
                    self.result.progress = 0
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.queue.task_done()
                time.sleep(0.05)
