import queue
import threading
import asyncio
import tempfile
import os
import uuid

recognizer = None
microphone = None
speech_queue = queue.Queue()
microphone_lock = threading.Lock()

def init_stt():
    global recognizer, microphone
    print("🎤 Đang khởi tạo Voice Recognition...")
    try:
        import speech_recognition as sr
        
        # List microphones
        print("\n📋 Danh sách microphones:")
        mic_list = sr.Microphone.list_microphone_names()
        
        if len(mic_list) == 0:
            print("❌ KHÔNG TÌM THẤY MICROPHONE!")
            return False
        
        for index, name in enumerate(mic_list):
            print(f"  [{index}] {name}")
        
        # Find best mic
        best_mic_index = None
        
        for index, name in enumerate(mic_list):
            name_lower = name.lower()
            if 'loopback' not in name_lower and 'stereo mix' not in name_lower:
                if any(kw in name_lower for kw in ['microphone', 'mic', 'headset', 'webcam', 'usb']):
                    best_mic_index = index
                    print(f"\n✅ Chọn: [{index}] {name}")
                    break
        
        if best_mic_index is None:
            best_mic_index = 0
            print(f"\n⚠️ Dùng mic mặc định: [{best_mic_index}] {mic_list[0]}")
        
        # Initialize
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.10
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.pause_threshold = 0.6
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.4
        
        microphone = sr.Microphone(device_index=best_mic_index)
        
        # Calibrate
        print("\n🔧 Đang hiệu chỉnh...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"   Energy threshold: {recognizer.energy_threshold:.0f}")

        print("\n🧪 Testing mic (say 'HELLO')...")
        try:
            with microphone as source:
                test_audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
                
                if test_audio:
                    size = len(test_audio.frame_data)
                    print(f"   ✅ Working! ({size}B)")
                    
                    if size < 1000:
                        print(f"   ⚠️ Weak signal - increase volume!")
                else:
                    print("   ⚠️ No audio captured")
                    
        except Exception as test_err:
            print(f"   ⚠️ Test failed: {test_err}")
            print("   → Mic might not work properly")
        
        print("✅ Voice Recognition ready!\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi init Voice: {e}")
        return False

def init_tts():
    """Initialize TTS"""
    print("🔊 Đang khởi tạo TTS...")
    try:
        import edge_tts
        import pygame
        
        # Test edge-tts version
        print(f"   📦 edge-tts version: {edge_tts.__version__ if hasattr(edge_tts, '__version__') else 'unknown'}")
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        
        print("✅ TTS ready!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   💡 Install: pip install edge-tts pygame")
        return False
    except Exception as e:
        print(f"⚠️ TTS init error: {e}")
        return False

def tts_worker():
    """TTS worker với error handling tốt hơn"""
    import edge_tts
    import pygame
    import tempfile
    import os
    import uuid
    import asyncio
    
    # Create event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def speak_with_retry(text, max_retries=2):
        """Speak với retry logic"""
        for attempt in range(max_retries):
            temp_file = None
            try:
                # Create temp file
                temp_file = os.path.join(
                    tempfile.gettempdir(), 
                    f"tts_{uuid.uuid4().hex[:8]}.mp3"
                )
                
                voices = [
                    "vi-VN-HoaiMyNeural",
                    "vi-VN-NamMinhNeural",
                    "en-US-AriaNeural"
                ]
                
                last_error = None
                
                for voice in voices:
                    try:
                        if attempt > 0:
                            print(f"   🔄 Retry {attempt+1}/{max_retries} with {voice}...")
                        
                        communicate = edge_tts.Communicate(
                            text, 
                            voice,
                            rate="+5%",
                        )
                        
                        await asyncio.wait_for(
                            communicate.save(temp_file),
                            timeout=10.0
                        )
                        
                        # Check file exists and not empty
                        if not os.path.exists(temp_file):
                            raise FileNotFoundError("TTS file not created")
                        
                        if os.path.getsize(temp_file) < 1000:
                            raise ValueError("TTS file too small")
                        
                        # Play audio
                        pygame.mixer.music.load(temp_file)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.05)
                        
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        
                        await asyncio.sleep(0.1)
                        
                        break
                        
                    except asyncio.TimeoutError:
                        last_error = f"Timeout with {voice}"
                        continue
                    except Exception as voice_err:
                        last_error = f"{voice}: {str(voice_err)}"
                        continue
                
                else:
                    if last_error:
                        raise Exception(f"All voices failed. Last: {last_error}")
                
                return
                
            except asyncio.TimeoutError:
                print(f"   ⏰ TTS timeout (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"   ❌ TTS failed after {max_retries} attempts: timeout")
                    
            except Exception as e:
                print(f"   ⚠️ TTS error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"   ❌ TTS failed after {max_retries} attempts")
                    
            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    # Main worker loop
    while True:
        text = speech_queue.get()
        
        if text is None:
            break
        
        try:
            loop.run_until_complete(speak_with_retry(text))
        except Exception as e:
            print(f"   ❌ TTS worker error: {e}")
        finally:
            speech_queue.task_done()
    
    loop.close()