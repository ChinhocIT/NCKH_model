import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForVision2Seq, AutoProcessor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR 8GB GPU
# ============================================================================

class Config:
    # Paths
    COCO_ROOT = './COCO_STUFF'
    
    TRAIN_IMG_DIR = f'{COCO_ROOT}/train2017'
    TRAIN_MASK_DIR = f'{COCO_ROOT}/stuffthingmaps_trainval2017/train2017'
    VAL_IMG_DIR = f'{COCO_ROOT}/val2017'
    VAL_MASK_DIR = f'{COCO_ROOT}/stuffthingmaps_trainval2017/val2017'
    
    NUM_CLASSES = 172  # COCO-Stuff: 171 classes + 1 unlabeled (0)
    
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './checkpoints'
    
    # Pre-extracted features directory
    FEATURES_DIR = './extracted_features'
    USE_PRE_EXTRACTED = True  # Set False to extract on-the-fly (slower)
    
    # Model - Qwen2-VL
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    VIT_FEATURES_DIM = 1536  # Qwen2-VL dimension
    DECODER_TYPE = 'pup'  # 'naive', 'pup', 'mla'
    
    # =========================================================
    # OPTIMIZED FOR V100 32GB GPU
    # =========================================================
    BATCH_SIZE = 8        # V100 can handle larger batches
    GRAD_ACCUM_STEPS = 2  # Effective batch = 8 * 2 = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4  # Lower LR for larger effective batch
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    
    # Data
    MAX_TRAIN_IMAGES = None
    MAX_VAL_IMAGES = None
    IMAGE_SIZE = 512      # Higher resolution for V100
    
    # =========================================================
    # V100 32GB OPTIMIZATION
    # =========================================================
    NUM_WORKERS = 4       # More workers for faster data loading
    PIN_MEMORY = True     # Pin memory for faster GPU transfer
    MIXED_PRECISION = True  # FP16 for V100 Tensor Cores (faster)
    USE_DICE_LOSS = True    # CE + Dice for best mIoU
    DICE_WEIGHT = 0.5       # lambda_dice = 0.5 (as in paper)
    
    # Logging
    PRINT_FREQ = 50
    SAVE_FREQ = 5
    EVAL_FREQ = 1
    EARLY_STOPPING_PATIENCE = 10
    
    # Resume training
    RESUME_TRAINING = True  # Auto-resume from checkpoint if available
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMPTY_CACHE_FREQ = 50  # Less frequent for V100 (more VRAM)

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# SETR DECODER (SAME AS MAIN CODE)
# ============================================================================

class SETRSegDecoder(nn.Module):
    """
    SETR-based Segmentation Decoder
    """
    
    def __init__(self, 
                 vit_features_dim=1536,
                 num_classes=172,
                 decoder_type='pup',
                 device="cuda",
                 debug=False):
        super().__init__()
        
        self.device = device
        self.debug = debug
        self.vit_features_dim = vit_features_dim
        
        # Feature adapter
        self.feature_adapter = nn.Sequential(
            nn.Linear(vit_features_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # SETR Decoder variants
        if decoder_type == 'naive':
            self.decoder = self._build_naive_decoder(num_classes)
        elif decoder_type == 'pup':
            self.decoder = self._build_pup_decoder(num_classes)
        elif decoder_type == 'mla':
            self.decoder = self._build_mla_decoder(num_classes)
        
        print(f"   🎨 SETR-{decoder_type.upper()} decoder initialized")
    
    def _build_naive_decoder(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def _build_pup_decoder(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def _build_mla_decoder(self, num_classes):
        self.aux_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        return nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, vit_features, target_size=None):
        B, N, C = vit_features.shape
        
        # Handle dimension mismatch
        if C != self.vit_features_dim:
            if not hasattr(self, 'dynamic_proj') or self.dynamic_proj is None:
                self.dynamic_proj = nn.Linear(C, self.vit_features_dim).to(vit_features.device)
            vit_features = self.dynamic_proj(vit_features)
        
        # Ensure float32
        if vit_features.dtype == torch.float16:
            vit_features = vit_features.float()
        
        # Adapt features
        adapted = self.feature_adapter(vit_features)  # (B, N, 768)
        
        # Reshape to 2D grid
        H_f = W_f = int(np.sqrt(N))
        if H_f * W_f != N:
            for h in range(int(np.sqrt(N)), 0, -1):
                if N % h == 0:
                    H_f = h
                    W_f = N // h
                    break
        
        adapted_2d = adapted.permute(0, 2, 1).reshape(B, 768, H_f, W_f)
        
        # Decode to segmentation map
        seg_map = self.decoder(adapted_2d)
        
        # Resize if needed
        if target_size is not None:
            seg_map = F.interpolate(
                seg_map, size=target_size,
                mode='bilinear', align_corners=True
            )
        
        return seg_map

# ============================================================================
# QWEN ViT ENCODER (FROM MAIN CODE)
# ============================================================================

class QwenViTEncoder:
    """
    Wrapper for Qwen2-VL's ViT encoder
    """
    
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
        print(f"\n📦 Loading Qwen2-VL: {model_name}")
        
        self.device = device
        
        # Load with FP16 for V100 Tensor Cores (faster than 8-bit quantization)
        # V100 32GB has enough VRAM, no need for quantization
        self.full_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 for V100
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Extract ViT encoder
        self.vit_encoder = self.full_model.visual
        self.vit_encoder.eval()
        
        # Freeze ViT
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        
        # Auto-detect dimensions
        self.vision_dim = None
        try:
            self.vision_dim = self.vit_encoder.config.hidden_size
        except:
            try:
                self.vision_dim = self.vit_encoder.config.embed_dim
            except:
                self.vision_dim = 1536  # Default
        
        print(f"✅ Qwen ViT loaded (dim: {self.vision_dim})")
        print(f"   Status: FROZEN ❄️")
    
    @torch.inference_mode()
    def extract_features(self, image_rgb: np.ndarray) -> torch.Tensor:
        """
        Extract ViT features from image (SAME AS MAIN CODE)
        """
        try:
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Process image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "Describe."}
                    ]
                }
            ]
            
            text_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            pixel_values = inputs['pixel_values']
            
            # Handle different shapes
            ndim = pixel_values.dim()
            
            if ndim == 2:
                features = pixel_values.unsqueeze(0)
                return features.float() if features.dtype == torch.float16 else features
            
            elif ndim == 3:
                return pixel_values.float() if pixel_values.dtype == torch.float16 else pixel_values
            
            elif ndim == 4:
                B, C, H, W = pixel_values.shape
            
            elif ndim == 5:
                B, num_imgs, C, H, W = pixel_values.shape
                pixel_values = pixel_values.reshape(B * num_imgs, C, H, W)
                B, C, H, W = pixel_values.shape
            
            else:
                return None
            
            # Calculate grid
            patch_size = getattr(self.vit_encoder.config, 'patch_size', 14)
            grid_h = H // patch_size
            grid_w = W // patch_size
            
            grid_thw = torch.tensor(
                [[1, grid_h, grid_w]], 
                device=self.device,
                dtype=torch.long
            )
            
            # Extract features
            vision_outputs = None
            strategies = [
                lambda: self.vit_encoder(pixel_values=pixel_values, grid_thw=grid_thw),
                lambda: self.vit_encoder(pixel_values),
            ]
            
            for strategy in strategies:
                try:
                    vision_outputs = strategy()
                    break
                except:
                    continue
            
            if vision_outputs is None:
                return None
            
            # Extract tensor
            features = None
            
            if isinstance(vision_outputs, tuple):
                for item in vision_outputs:
                    if isinstance(item, torch.Tensor) and item.dim() == 3:
                        features = item
                        break
            
            elif hasattr(vision_outputs, 'last_hidden_state'):
                features = vision_outputs.last_hidden_state
            
            elif isinstance(vision_outputs, torch.Tensor):
                features = vision_outputs
            
            if features is None:
                return None
            
            # Ensure 3D: (B, N, C)
            if features.dim() == 4:
                B, C, H_f, W_f = features.shape
                features = features.flatten(2).permute(0, 2, 1)
            elif features.dim() == 2:
                features = features.unsqueeze(0)
            
            if features.dtype == torch.float16:
                features = features.float()
            
            return features
            
        except Exception as e:
            print(f"❌ Extract error: {e}")
            return None

# ============================================================================
# COCO-STUFF DATASET (MODIFIED FOR QWEN)
# ============================================================================

class COCOStuffQwenDataset(Dataset):
    """
    COCO-Stuff Dataset with Qwen ViT preprocessing
    """
    
    def __init__(self, img_dir, mask_dir, qwen_encoder, 
                 max_images=None, image_size=512, augment=False):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.qwen_encoder = qwen_encoder
        self.image_size = image_size
        self.augment = augment
        
        print(f"\n📦 Loading COCO-Stuff dataset...")
        print(f"   Images: {img_dir}")
        print(f"   Masks:  {mask_dir}")
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"❌ Directory not found!")
            self.samples = []
            return
        
        # Find matching pairs
        image_files = [f for f in os.listdir(img_dir) 
                      if f.endswith('.jpg') or f.endswith('.jpeg')]
        
        self.samples = []
        for img_filename in tqdm(sorted(image_files), desc='Matching pairs'):
            base_name = os.path.splitext(img_filename)[0]
            mask_filename = base_name + '.png'
            
            img_path = os.path.join(img_dir, img_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                self.samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'filename': img_filename
                })
                
                if max_images and len(self.samples) >= max_images:
                    break
        
        print(f"✅ Loaded {len(self.samples):,} pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image with error handling
        img = cv2.imread(sample['img_path'])
        if img is None:
            # Return dummy data if image is corrupted
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask with error handling
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = np.clip(mask, 0, 171)
        
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.8, 1.2)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size), 
                        interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_NEAREST)
        
        # Extract Qwen ViT features
        vit_features = self.qwen_encoder.extract_features(img)
        
        if vit_features is None:
            # Fallback: return dummy features
            vit_features = torch.randn(1, 256, 1536)
        
        # Remove batch dimension if present and move to CPU
        if vit_features.dim() == 3 and vit_features.size(0) == 1:
            vit_features = vit_features.squeeze(0)
        
        # IMPORTANT: Move to CPU for DataLoader compatibility
        vit_features = vit_features.cpu()
        
        mask = torch.from_numpy(mask).long()
        
        return vit_features, mask

# ============================================================================
# PRE-EXTRACTED FEATURES DATASET (FAST!)
# ============================================================================

class PreExtractedDataset(Dataset):
    """
    Dataset that loads pre-extracted ViT features from disk.
    Much faster than extracting features on-the-fly!
    
    NOTE: Augmentation is NOT supported! Pre-extracted features cannot be
    spatially transformed to match augmented masks.
    """
    
    def __init__(self, features_dir, mask_dir, image_size=512, max_images=None):
        self.features_dir = features_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        print(f"\n📦 Loading Pre-Extracted Dataset...")
        print(f"   Features: {features_dir}")
        print(f"   Masks:    {mask_dir}")
        
        if not os.path.exists(features_dir):
            print(f"❌ Features directory not found! Run pre-extraction first.")
            self.samples = []
            return
        
        # Find all extracted feature files
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]
        
        self.samples = []
        for feat_file in tqdm(sorted(feature_files), desc='Loading features list'):
            base_name = os.path.splitext(feat_file)[0]  # e.g., "000000000001"
            mask_filename = base_name + '.png'
            
            feat_path = os.path.join(features_dir, feat_file)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                self.samples.append({
                    'feat_path': feat_path,
                    'mask_path': mask_path,
                    'filename': base_name
                })
                
                # Limit number of samples if specified
                if max_images and len(self.samples) >= max_images:
                    break
        
        print(f"✅ Loaded {len(self.samples):,} pre-extracted samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pre-extracted features (FAST!)
        vit_features = torch.load(sample['feat_path'], map_location='cpu')
        
        # Ensure correct shape (N, C) without batch dim
        if vit_features.dim() == 3:
            vit_features = vit_features.squeeze(0)
        
        # Load mask with error handling
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None:
            # Return zero mask if file is corrupted
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = np.clip(mask, 0, 171)
            # Resize mask to target size
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_NEAREST)
        
        # NOTE: Cannot augment pre-extracted features!
        # Flipping mask without flipping features would cause mismatch.
        # If augmentation is needed, use COCOStuffQwenDataset instead.
        
        mask = torch.from_numpy(mask).long()
        
        return vit_features, mask


def pre_extract_features(config, split='train'):
    """
    Pre-extract all ViT features and save to disk.
    Run this ONCE before training!
    
    Args:
        config: Config object
        split: 'train' or 'val'
    """
    import time
    
    if split == 'train':
        img_dir = config.TRAIN_IMG_DIR
        output_dir = os.path.join(config.FEATURES_DIR, 'train')
    else:
        img_dir = config.VAL_IMG_DIR
        output_dir = os.path.join(config.FEATURES_DIR, 'val')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 PRE-EXTRACTING FEATURES: {split.upper()}")
    print(f"{'='*70}")
    print(f"   Source: {img_dir}")
    print(f"   Output: {output_dir}")
    
    # Check how many already extracted
    existing = set(f.replace('.pt', '') for f in os.listdir(output_dir) if f.endswith('.pt'))
    print(f"   Already extracted: {len(existing):,}")
    
    # Load Qwen encoder
    qwen_encoder = QwenViTEncoder(
        model_name=config.QWEN_MODEL,
        device=config.DEVICE
    )
    
    # Get all images
    image_files = sorted([f for f in os.listdir(img_dir) 
                         if f.endswith('.jpg') or f.endswith('.jpeg')])
    
    # Filter out already extracted
    to_extract = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        if base_name not in existing:
            to_extract.append(img_file)
    
    print(f"   To extract: {len(to_extract):,}")
    
    if len(to_extract) == 0:
        print(f"\n✅ All features already extracted!")
        return
    
    # Extract features
    start_time = time.time()
    success = 0
    failed = 0
    
    pbar = tqdm(to_extract, desc=f'Extracting {split}')
    
    for img_file in pbar:
        try:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            output_path = os.path.join(output_dir, base_name + '.pt')
            
            # Load and resize image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR)
            
            # Extract features
            features = qwen_encoder.extract_features(img)
            
            if features is not None:
                # Save to disk (CPU tensor to save space)
                torch.save(features.cpu(), output_path)
                success += 1
            else:
                failed += 1
            
            # Update progress
            pbar.set_postfix({'success': success, 'failed': failed})
            
            # Clear cache periodically
            if (success + failed) % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            failed += 1
            continue
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE: {split.upper()}")
    print(f"{'='*70}")
    print(f"   Success: {success:,}")
    print(f"   Failed:  {failed:,}")
    print(f"   Time:    {elapsed/3600:.2f} hours")
    print(f"   Speed:   {success/elapsed:.2f} images/sec")
    print(f"   Saved to: {output_dir}")
    
    # Cleanup
    del qwen_encoder
    torch.cuda.empty_cache()


# ============================================================================
# LOSS & METRICS (SAME AS BEFORE)
# ============================================================================

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=172, ignore_index=255, use_dice=False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.use_dice = use_dice
        self.dice_weight = 0.5 if use_dice else 0.0
    
    def dice_loss(self, pred, target):
        # Memory-efficient dice loss - compute per class separately
        pred = F.softmax(pred, dim=1)
        dice_sum = 0.0
        valid_classes = 0
        
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            if union > 0:
                dice_sum += (2.0 * intersection + 1e-7) / (union + 1e-7)
                valid_classes += 1
        
        if valid_classes > 0:
            return 1.0 - dice_sum / valid_classes
        return torch.tensor(0.0, device=pred.device)
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        if self.use_dice:
            dice = self.dice_loss(pred, target)
            return ce + self.dice_weight * dice
        return ce

class SegmentationMetrics:
    def __init__(self, num_classes=172, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.total_correct = 0
        self.total_pixels = 0
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
    
    def update(self, pred, target):
        pred = pred.argmax(dim=1).cpu().numpy()
        target = target.cpu().numpy()
        
        valid = (target != self.ignore_index)
        self.total_correct += (pred[valid] == target[valid]).sum()
        self.total_pixels += valid.sum()
        
        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            self.intersection[cls] += (pred_cls & target_cls & valid).sum()
            self.union[cls] += ((pred_cls | target_cls) & valid).sum()
    
    def get_metrics(self):
        pixel_acc = self.total_correct / (self.total_pixels + 1e-10)
        iou = self.intersection / (self.union + 1e-10)
        mean_iou = np.nanmean(iou)
        return {'pixel_acc': pixel_acc, 'mean_iou': mean_iou}

# ============================================================================
# TRAINER (MODIFIED FOR QWEN)
# ============================================================================

class SETRQwenTrainer:
    def __init__(self, config, load_encoder=True):
        self.config = config
        self.device = config.DEVICE
        
        print(f"\n{'='*70}")
        print(f"🚀 SETR TRAINER WITH QWEN ViT")
        print(f"{'='*70}")
        
        # Only load Qwen ViT if needed (not using pre-extracted features)
        if load_encoder and not getattr(config, 'USE_PRE_EXTRACTED', False):
            self.qwen_encoder = QwenViTEncoder(
                model_name=config.QWEN_MODEL,
                device=self.device
            )
            print("   ❄️  Qwen ViT: LOADED & FROZEN")
        else:
            self.qwen_encoder = None
            print("   ⚡ Using PRE-EXTRACTED features (Qwen ViT not loaded)")
        
        # Initialize decoder
        self.seg_decoder = SETRSegDecoder(
            vit_features_dim=config.VIT_FEATURES_DIM,
            num_classes=config.NUM_CLASSES,
            decoder_type=config.DECODER_TYPE,
            device=self.device
        ).to(self.device)
        
        print("   🔥 Decoder: TRAINABLE")
        
        self.criterion = SegmentationLoss(
            config.NUM_CLASSES, 
            use_dice=getattr(config, 'USE_DICE_LOSS', False)
        ).to(self.device)
        self.optimizer = AdamW(
            self.seg_decoder.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        self.scaler = torch.amp.GradScaler('cuda') if config.MIXED_PRECISION else None
        
        self.train_metrics = SegmentationMetrics(config.NUM_CLASSES)
        self.val_metrics = SegmentationMetrics(config.NUM_CLASSES)
        
        self.history = {
            'train_loss': [], 'train_miou': [],
            'val_loss': [], 'val_miou': [], 'lr': []
        }
        self.best_miou = 0.0
        self.start_epoch = 0
        self.patience_counter = 0
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint to resume training."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'setr_qwen_latest.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"   ⚠️ No checkpoint found at {checkpoint_path}")
            return False
        
        print(f"\n📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.seg_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_miou = checkpoint['best_miou']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"   ✅ Resumed from epoch {self.start_epoch}")
        print(f"   📊 Best mIoU so far: {self.best_miou:.4f}")
        return True
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': self.seg_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'history': self.history,
            'patience_counter': self.patience_counter
        }
        
        latest = os.path.join(self.config.CHECKPOINT_DIR, 'setr_qwen_latest.pth')
        torch.save(checkpoint, latest)
        print(f"   💾 Saved: setr_qwen_latest.pth")
        
        if is_best:
            best = os.path.join(self.config.CHECKPOINT_DIR, 'setr_qwen_best.pth')
            torch.save(checkpoint, best)
            print(f"   🏆 Saved: setr_qwen_best.pth")
    
    def train_epoch(self, dataloader, epoch):
        self.seg_decoder.train()
        self.train_metrics.reset()
        total_loss = 0.0
        accum_steps = getattr(self.config, 'GRAD_ACCUM_STEPS', 1)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        self.optimizer.zero_grad()  # Zero grad at start
        
        for batch_idx, (vit_features, masks) in enumerate(pbar):
            vit_features = vit_features.to(self.device)
            masks = masks.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.config.MIXED_PRECISION):
                seg_logits = self.seg_decoder(
                    vit_features, 
                    target_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                )
                loss = self.criterion(seg_logits, masks)
                loss = loss / accum_steps  # Scale loss for accumulation
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.seg_decoder.parameters(), self.config.GRAD_CLIP
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.seg_decoder.parameters(), self.config.GRAD_CLIP
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accum_steps
            self.train_metrics.update(seg_logits.detach(), masks)
            
            if (batch_idx + 1) % 10 == 0:
                metrics = self.train_metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{loss.item() * accum_steps:.4f}",
                    'mIoU': f"{metrics['mean_iou']:.4f}"
                })
            
            # Clear cache frequently for 8GB GPU
            if (batch_idx + 1) % self.config.EMPTY_CACHE_FREQ == 0:
                torch.cuda.empty_cache()
            
            # Delete tensors to free memory
            del seg_logits, loss
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.train_metrics.get_metrics()
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_miou'].append(metrics['mean_iou'])
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.seg_decoder.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        
        for vit_features, masks in tqdm(dataloader, desc="Validation"):
            vit_features = vit_features.to(self.device)
            masks = masks.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.config.MIXED_PRECISION):
                seg_logits = self.seg_decoder(
                    vit_features,
                    target_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                )
                loss = self.criterion(seg_logits, masks)
            
            total_loss += loss.item()
            self.val_metrics.update(seg_logits, masks)
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.val_metrics.get_metrics()
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_miou'].append(metrics['mean_iou'])
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader):
        print(f"\n{'='*70}")
        print(f"🎯 STARTING TRAINING")
        print(f"{'='*70}\n")
        
        self.current_epoch = self.start_epoch  # Track current epoch for emergency save
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch  # Update current epoch
            
            print(f"\n📅 Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("─" * 70)
            
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            if (epoch + 1) % self.config.EVAL_FREQ == 0:
                val_loss, val_metrics = self.validate(val_loader)
            else:
                val_loss, val_metrics = 0, {'mean_iou': 0, 'pixel_acc': 0}
            
            self.scheduler.step()
            
            print(f"\n📊 Summary:")
            print(f"   Train - Loss: {train_loss:.4f} | mIoU: {train_metrics['mean_iou']:.4f}")
            if (epoch + 1) % self.config.EVAL_FREQ == 0:
                print(f"   Val   - Loss: {val_loss:.4f} | mIoU: {val_metrics['mean_iou']:.4f}")
            
            is_best = False
            if (epoch + 1) % self.config.EVAL_FREQ == 0:
                if val_metrics['mean_iou'] > self.best_miou:
                    self.best_miou = val_metrics['mean_iou']
                    is_best = True
                    self.patience_counter = 0
                    print(f"   🏆 New best: {self.best_miou:.4f}")
                else:
                    self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping!")
                break
        
        print(f"\n✅ DONE! Best mIoU: {self.best_miou:.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(f"🚀 SETR TRAINING WITH QWEN ViT")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU detected!")
    
    # =========================================================
    # STEP 1: PRE-EXTRACT FEATURES (IF NEEDED)
    # =========================================================
    if getattr(Config, 'USE_PRE_EXTRACTED', True):
        train_features_dir = os.path.join(Config.FEATURES_DIR, 'train')
        val_features_dir = os.path.join(Config.FEATURES_DIR, 'val')
        
        # Check if extraction is needed
        train_exists = os.path.exists(train_features_dir) and len(os.listdir(train_features_dir)) > 0
        val_exists = os.path.exists(val_features_dir) and len(os.listdir(val_features_dir)) > 0
        
        if not train_exists or not val_exists:
            print(f"\n{'='*70}")
            print("📥 PRE-EXTRACTING FEATURES (ONE-TIME)")
            print("="*70)
            print("   This will take ~3-5 hours on V100, but only needs to run ONCE!")
            print("   Features will be saved and reused for all future training.\n")
            
            if not train_exists:
                pre_extract_features(Config, split='train')
            if not val_exists:
                pre_extract_features(Config, split='val')
            
            print(f"\n✅ Pre-extraction complete! Starting training...\n")
        else:
            print(f"\n✅ Pre-extracted features found:")
            print(f"   Train: {train_features_dir}")
            print(f"   Val:   {val_features_dir}")
    
    # =========================================================
    # STEP 2: INITIALIZE TRAINER
    # =========================================================
    trainer = SETRQwenTrainer(Config)
    
    # Auto-resume from checkpoint if available
    if getattr(Config, 'RESUME_TRAINING', True):
        trainer.load_checkpoint()
    
    # =========================================================
    # STEP 3: CREATE DATASETS
    # =========================================================
    print(f"\n{'='*70}")
    print("📦 CREATING DATASETS")
    print("="*70)
    
    if getattr(Config, 'USE_PRE_EXTRACTED', True):
        # Use fast pre-extracted dataset
        # NOTE: No augmentation possible with pre-extracted features!
        print("   Mode: PRE-EXTRACTED FEATURES (FAST! 🚀)")
        print("   ⚠️  Augmentation disabled (pre-extracted features cannot be augmented)")
        
        train_dataset = PreExtractedDataset(
            features_dir=os.path.join(Config.FEATURES_DIR, 'train'),
            mask_dir=Config.TRAIN_MASK_DIR,
            image_size=Config.IMAGE_SIZE,
            max_images=Config.MAX_TRAIN_IMAGES
        )
        val_dataset = PreExtractedDataset(
            features_dir=os.path.join(Config.FEATURES_DIR, 'val'),
            mask_dir=Config.VAL_MASK_DIR,
            image_size=Config.IMAGE_SIZE,
            max_images=Config.MAX_VAL_IMAGES
        )
    else:
        # Use on-the-fly extraction (slow)
        print("   Mode: ON-THE-FLY EXTRACTION (SLOW ⚠️)")
        
        train_dataset = COCOStuffQwenDataset(
            img_dir=Config.TRAIN_IMG_DIR,
            mask_dir=Config.TRAIN_MASK_DIR,
            qwen_encoder=trainer.qwen_encoder,
            max_images=Config.MAX_TRAIN_IMAGES,
            image_size=Config.IMAGE_SIZE,
            augment=True
        )
        val_dataset = COCOStuffQwenDataset(
            img_dir=Config.VAL_IMG_DIR,
            mask_dir=Config.VAL_MASK_DIR,
            qwen_encoder=trainer.qwen_encoder,
            max_images=Config.MAX_VAL_IMAGES,
            image_size=Config.IMAGE_SIZE,
            augment=False
        )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n❌ No data loaded! Check your paths.")
        return
    
    print(f"\n{'='*70}")
    print("📦 CREATING DATALOADERS")
    print("="*70)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False
    )
    
    print(f"\n✅ Dataloaders ready:")
    print(f"   Train: {len(train_dataset):,} images → {len(train_loader):,} batches")
    print(f"   Val:   {len(val_dataset):,} images → {len(val_loader):,} batches")
    
    if getattr(Config, 'USE_PRE_EXTRACTED', True):
        print(f"\n⏱️  Estimated time per epoch: ~10-15 minutes (FAST!)")
    else:
        print(f"\n⏱️  Estimated time per epoch:")
        print(f"   ~{len(train_loader) * Config.BATCH_SIZE * 2 / 60:.1f} minutes")
        print(f"   (Qwen feature extraction is slower than SimpleViT)")
    
    print(f"\n{'='*70}")
    print("⚠️  READY TO START TRAINING")
    print("="*70)
    print(f"📊 Configuration:")
    print(f"   Model: {Config.QWEN_MODEL}")
    print(f"   Decoder: SETR-{Config.DECODER_TYPE.upper()}")
    print(f"   Total train images: {len(train_dataset):,}")
    print(f"   Total epochs: {Config.NUM_EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print(f"\n🔥 Training strategy:")
    if getattr(Config, 'USE_PRE_EXTRACTED', True):
        print(f"   ⚡ PRE-EXTRACTED FEATURES (Fast mode!)")
    else:
        print(f"   ❄️  FROZEN: Qwen ViT (FP16)")
    print(f"   🔥 TRAINABLE: SETR Decoder only")
    print(f"   📦 Gradient Accumulation: {Config.GRAD_ACCUM_STEPS} steps")
    print(f"   🎯 Effective Batch Size: {Config.BATCH_SIZE * Config.GRAD_ACCUM_STEPS}")
    if getattr(Config, 'USE_PRE_EXTRACTED', True):
        print(f"\n💡 Note: Features are pre-extracted, so training is FAST!")
    print(f"\n{'='*70}")
    print("🚀 STARTING TRAINING...")
    print("="*70 + "\n")
    
    try:
        trainer.train(train_loader, val_loader)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
        print("💾 Saving checkpoint...")
        epoch = getattr(trainer, 'current_epoch', trainer.start_epoch)
        trainer.save_checkpoint(epoch)
        print("✅ Checkpoint saved!")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            print("\n💾 Attempting emergency save...")
            epoch = getattr(trainer, 'current_epoch', trainer.start_epoch)
            trainer.save_checkpoint(epoch)
            print("✅ Emergency checkpoint saved!")
        except:
            pass
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Best mIoU: {trainer.best_miou:.4f}")
    print(f"   Total epochs: {len(trainer.history['train_loss'])}")
    print(f"\n💾 Checkpoints saved to: {Config.CHECKPOINT_DIR}")
    
    # Generate training plots
    plot_training_curves(trainer.history, Config.OUTPUT_DIR)


def plot_training_curves(history, output_dir):
    """
    Generate scientific training plots for paper/report.
    
    Plots:
    1. Loss curves (train vs val)
    2. mIoU curves (train vs val)
    3. Learning rate schedule
    4. Summary figure with all metrics
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    if len(epochs) == 0:
        print("⚠️ No training history to plot")
        return
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'train': '#2196F3', 'val': '#F44336', 'lr': '#4CAF50'}
    
    # =========================================================
    # Figure 1: Loss Curves
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['train_loss'], 'o-', color=colors['train'], 
            linewidth=2, markersize=6, label='Train Loss')
    if history['val_loss'] and len(history['val_loss']) == len(epochs):
        ax.plot(epochs, history['val_loss'], 's-', color=colors['val'], 
                linewidth=2, markersize=6, label='Val Loss')
        
        # Mark best val loss
        best_idx = np.argmin(history['val_loss'])
        ax.annotate(f'Best: {history["val_loss"][best_idx]:.4f}',
                   xy=(epochs[best_idx], history['val_loss'][best_idx]),
                   xytext=(epochs[best_idx] + 1, history['val_loss'][best_idx] + 0.1),
                   fontsize=10, color=colors['val'],
                   arrowprops=dict(arrowstyle='->', color=colors['val']))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (CE + Dice)', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved: {save_path}")
    
    # =========================================================
    # Figure 2: mIoU Curves
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history['train_miou'], 'o-', color=colors['train'], 
            linewidth=2, markersize=6, label='Train mIoU')
    if history['val_miou'] and len(history['val_miou']) == len(epochs):
        ax.plot(epochs, history['val_miou'], 's-', color=colors['val'], 
                linewidth=2, markersize=6, label='Val mIoU')
        
        # Mark best val mIoU
        best_idx = np.argmax(history['val_miou'])
        best_miou = history['val_miou'][best_idx]
        ax.annotate(f'Best: {best_miou:.4f}',
                   xy=(epochs[best_idx], best_miou),
                   xytext=(epochs[best_idx] + 1, best_miou - 0.02),
                   fontsize=10, color=colors['val'],
                   arrowprops=dict(arrowstyle='->', color=colors['val']))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Training & Validation mIoU', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'miou_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved: {save_path}")
    
    # =========================================================
    # Figure 3: Learning Rate Schedule
    # =========================================================
    if history['lr']:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs[:len(history['lr'])], history['lr'], '-', 
                color=colors['lr'], linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'learning_rate.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved: {save_path}")
    
    # =========================================================
    # Figure 4: Summary (2x2 grid)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'o-', color=colors['train'], label='Train')
    if history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 's-', color=colors['val'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU
    axes[0, 1].plot(epochs, history['train_miou'], 'o-', color=colors['train'], label='Train')
    if history['val_miou']:
        axes[0, 1].plot(epochs, history['val_miou'], 's-', color=colors['val'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('mIoU Curves', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if history['lr']:
        axes[1, 0].plot(epochs[:len(history['lr'])], history['lr'], '-', color=colors['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Summary stats
    axes[1, 1].axis('off')
    if history['val_miou']:
        best_epoch = np.argmax(history['val_miou']) + 1
        best_miou = max(history['val_miou'])
        final_loss = history['val_loss'][-1] if history['val_loss'] else history['train_loss'][-1]
    else:
        best_epoch = len(epochs)
        best_miou = max(history['train_miou']) if history['train_miou'] else 0
        final_loss = history['train_loss'][-1]
    
    stats_text = f"""
    Training Summary
    {'─' * 30}
    
    Total Epochs: {len(epochs)}
    Best Epoch: {best_epoch}
    
    Best mIoU: {best_miou:.4f}
    Final Loss: {final_loss:.4f}
    
    Loss Function: CE + Dice (λ=0.5)
    Optimizer: AdamW
    Scheduler: CosineAnnealingWarmRestarts
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('SETR Decoder Training Report', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {save_path}")
    
    # Save history to JSON
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"💾 Saved: {history_path}")
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

