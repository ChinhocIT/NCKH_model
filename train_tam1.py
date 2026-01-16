import os
import sys
import json
import signal
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_multiprocessing():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

def test_cuda():
    print(f"   PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        return True
    print("   ❌ CUDA not available")
    return False

class Config:
    COCO_ROOT = './COCO_STUFF'
    TRAIN_IMG_DIR = f'{COCO_ROOT}/train2017'
    TRAIN_MASK_DIR = f'{COCO_ROOT}/stuffthingmaps_trainval2017/train2017'
    VAL_IMG_DIR = f'{COCO_ROOT}/val2017'
    VAL_MASK_DIR = f'{COCO_ROOT}/stuffthingmaps_trainval2017/val2017'
    NUM_CLASSES = 172
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './checkpoints'
    FEATURES_DIR = './extracted_features'
    USE_PRE_EXTRACTED = True 
    DINO_VARIANT = 'vitb14' 
    VIT_FEATURES_DIM = 768 
    DECODER_TYPE = 'pup' 
    IMAGE_SIZE = 518
    BATCH_SIZE = 128
    GRAD_ACCUM_STEPS = 1      
    NUM_WORKERS = 16
    NUM_EPOCHS = 30           
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    MAX_TRAIN_IMAGES = None
    MAX_VAL_IMAGES = None
    PIN_MEMORY = True 
    PREFETCH_FACTOR = 4 
    MIXED_PRECISION = True 
    USE_DICE_LOSS = True
    DICE_WEIGHT = 0.5
    DICE_START_EPOCH = 10
    PRINT_FREQ = 50
    SAVE_FREQ = 5
    EVAL_FREQ = 1
    EARLY_STOPPING_PATIENCE = 10
    RESUME_TRAINING = True
    DEVICE = 'cuda'
    EMPTY_CACHE_FREQ = 500

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class SETRSegDecoder(nn.Module):
    def __init__(self, vit_features_dim=768, num_classes=172, decoder_type='pup', device="cuda"):
        super().__init__()
        self.device = device
        self.vit_features_dim = vit_features_dim
        self.feature_adapter = nn.Sequential(
            nn.Linear(vit_features_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        if decoder_type == 'naive':
            self.decoder = self._build_naive_decoder(num_classes)
        elif decoder_type == 'pup':
            self.decoder = self._build_pup_decoder(num_classes)
        elif decoder_type == 'mla':
            self.decoder = self._build_mla_decoder(num_classes)
        print(f"   🎨 SETR-{decoder_type.upper()} initialized")
    
    def _build_naive_decoder(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(768, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def _build_pup_decoder(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def _build_mla_decoder(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(768, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, vit_features, target_size=None):
        B, N, C = vit_features.shape
        if vit_features.dtype == torch.float16:
            vit_features = vit_features.float()
        adapted = self.feature_adapter(vit_features)
        H_f = W_f = int(N ** 0.5)
        adapted_2d = adapted.permute(0, 2, 1).reshape(B, 768, H_f, W_f)
        seg_map = self.decoder(adapted_2d)
        if target_size is not None:
            seg_map = F.interpolate(seg_map, size=target_size, mode='bilinear', align_corners=True)
        return seg_map

class DINOv2Encoder:
    def __init__(self, variant='vitb14', device="cuda"):
        print(f"\n📦 Loading DINOv2 {variant.upper()}...")
        self.device = device
        self.patch_size = 14
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{variant}', pretrained=True)
        except:
            self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{variant}', pretrained=True, trust_repo=True)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"✅ DINOv2 loaded & frozen")
    
    @torch.inference_mode()
    def extract_features(self, image_rgb):
        from PIL import Image
        import torchvision.transforms as T
        if isinstance(image_rgb, np.ndarray):
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        else:
            pil_image = image_rgb
        transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(pil_image).unsqueeze(0).to(self.device)
        features = self.model.forward_features(pixel_values)
        if isinstance(features, dict):
            return features['x_norm_patchtokens']
        return features[:, 1:]

class PreExtractedDataset(Dataset):
    def __init__(self, features_dir, mask_dir, image_size=518, max_images=None,
                 augment=True, noise_std=0.02, dropout_p=0.1):
        self.features_dir = features_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]
        self.samples = []
        for f in feature_files:
            base = os.path.splitext(f)[0]
            mask_p = os.path.join(mask_dir, base + '.png')
            if os.path.exists(mask_p):
                self.samples.append({'feat_path': os.path.join(features_dir, f), 'mask_path': mask_p})
            if max_images and len(self.samples) >= max_images:
                break
        print(f"✅ Loaded {len(self.samples)} samples (Augment={augment})")

    def __len__(self):
        return len(self.samples)

    def _augment_features(self, features):
        if not self.augment:
            return features
        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std
        if self.dropout_p > 0 and np.random.rand() > 0.5:
            mask = torch.rand_like(features) > self.dropout_p
            features = features * mask / (1 - self.dropout_p)
        if np.random.rand() > 0.9:
            features = features * np.random.uniform(0.95, 1.05)
        return features

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vit_features = torch.load(sample['feat_path'], map_location='cpu', weights_only=True).float()
        if vit_features.dim() == 3:
            vit_features = vit_features.squeeze(0)
        vit_features = self._augment_features(vit_features)
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask = torch.from_numpy(np.clip(mask, 0, 171)).long()
        return vit_features, mask

def pre_extract_features(config, split='train'):
    if split == 'train':
        img_dir, out_dir = config.TRAIN_IMG_DIR, os.path.join(config.FEATURES_DIR, 'train')
    else:
        img_dir, out_dir = config.VAL_IMG_DIR, os.path.join(config.FEATURES_DIR, 'val')
    os.makedirs(out_dir, exist_ok=True)
    existing = set(f.replace('.pt', '') for f in os.listdir(out_dir) if f.endswith('.pt'))
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg')) and os.path.splitext(f)[0] not in existing]
    if not image_files:
        return
    print(f"🚀 Extracting {len(image_files)} images to {out_dir}...")
    dino = DINOv2Encoder(config.DINO_VARIANT, config.DEVICE)
    for img_file in tqdm(image_files):
        try:
            img = cv2.imread(os.path.join(img_dir, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feats = dino.extract_features(img)
            torch.save(feats.half().cpu(), os.path.join(out_dir, os.path.splitext(img_file)[0] + '.pt'))
        except:
            pass
    del dino
    torch.cuda.empty_cache()

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, use_dice=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.use_dice = use_dice
        self.dice_weight = 0.5
    
    def set_dice_enabled(self, enabled):
        self.use_dice = enabled
    
    def forward(self, pred, target):
        loss = self.ce(pred, target)
        if self.use_dice:
            pred_s = F.softmax(pred, dim=1)
            target_1h = F.one_hot(target.clamp(0, 171), 172).permute(0, 3, 1, 2).float()
            valid = (target != 255).unsqueeze(1)
            inter = (pred_s * target_1h * valid).sum(dim=(2, 3))
            union = (pred_s * valid).sum(dim=(2, 3)) + (target_1h * valid).sum(dim=(2, 3))
            dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
            loss += self.dice_weight * dice.mean()
        return loss

class SETRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.decoder = SETRSegDecoder(config.VIT_FEATURES_DIM, config.NUM_CLASSES, config.DECODER_TYPE, self.device).to(self.device)
        self.opt = AdamW(self.decoder.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.sched = CosineAnnealingWarmRestarts(self.opt, T_0=5, T_mult=2)
        self.scaler = torch.amp.GradScaler('cuda') if config.MIXED_PRECISION else None
        self.criterion = SegmentationLoss(config.NUM_CLASSES, False).to(self.device)
        self.best_miou = 0.0
        self.start_epoch = 0
        self.history = {'train_loss': [], 'val_miou': [], 'lr': []}

    def load_checkpoint(self):
        path = os.path.join(self.config.CHECKPOINT_DIR, 'setr_dino_best.pth')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.decoder.load_state_dict(ckpt['state'], strict=False)
            self.best_miou = ckpt.get('best', 0)
            self.start_epoch = ckpt.get('epoch', 0) + 1
            print(f"📥 Resumed from epoch {self.start_epoch}, best mIoU: {self.best_miou:.4f}")

    def save_checkpoint(self, epoch, miou):
        ckpt = {'state': self.decoder.state_dict(), 'best': miou, 'epoch': epoch}
        torch.save(ckpt, os.path.join(self.config.CHECKPOINT_DIR, 'setr_dino_latest.pth'))
        if miou >= self.best_miou:
            self.best_miou = miou
            torch.save(ckpt, os.path.join(self.config.CHECKPOINT_DIR, 'setr_dino_best.pth'))
            print("🏆 New Best!")

    def train_epoch(self, loader, epoch):
        self.decoder.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{self.config.NUM_EPOCHS}")
        for feat, mask in pbar:
            feat = feat.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=self.config.MIXED_PRECISION):
                pred = self.decoder(feat, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
                loss = self.criterion(pred, mask)
            self.opt.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.GRAD_CLIP)
                self.opt.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.decoder.eval()
        inter_sum, union_sum = np.zeros(172), np.zeros(172)
        for feat, mask in tqdm(loader, desc="Val"):
            feat = feat.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=self.config.MIXED_PRECISION):
                pred = self.decoder(feat, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            p = pred.argmax(1).cpu().numpy()
            t = mask.cpu().numpy()
            valid = t != 255
            for c in range(172):
                inter_sum[c] += np.sum((p == c) & (t == c) & valid)
                union_sum[c] += np.sum(((p == c) | (t == c)) & valid)
        iou = inter_sum / (union_sum + 1e-10)
        return np.nanmean(iou)

    def run(self, train_dl, val_dl):
        print(f"\n🚀 Training on {self.device} for {self.config.NUM_EPOCHS} epochs...")
        for ep in range(self.start_epoch, self.config.NUM_EPOCHS):
            if ep == self.config.DICE_START_EPOCH:
                print(f"   🎲 Enabling Dice Loss from epoch {ep+1}")
                self.criterion.set_dice_enabled(True)
            loss = self.train_epoch(train_dl, ep)
            miou = self.validate(val_dl) if (ep + 1) % self.config.EVAL_FREQ == 0 else 0
            self.history['train_loss'].append(loss)
            self.history['val_miou'].append(miou)
            self.history['lr'].append(self.opt.param_groups[0]['lr'])
            self.sched.step()
            print(f"📊 Ep {ep+1}: Loss {loss:.4f} | mIoU {miou:.4f}")
            self.save_checkpoint(ep, miou)
        self.plot_curves()
        with open(f"{self.config.OUTPUT_DIR}/history.json", 'w') as f:
            json.dump(self.history, f)
        print(f"\n✅ Done! Best mIoU: {self.best_miou:.4f}")

    def plot_curves(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
        ax2.plot(self.history['val_miou'], 'r-', label='Val mIoU')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('mIoU'); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.config.OUTPUT_DIR}/training_curves.png", dpi=150)
        plt.close()
        print(f"📈 Saved: {self.config.OUTPUT_DIR}/training_curves.png")

def main():
    setup_multiprocessing()
    if not test_cuda():
        return
    if Config.USE_PRE_EXTRACTED:
        pre_extract_features(Config, 'train')
        pre_extract_features(Config, 'val')
    train_ds = PreExtractedDataset(f"{Config.FEATURES_DIR}/train", Config.TRAIN_MASK_DIR, augment=True)
    val_ds = PreExtractedDataset(f"{Config.FEATURES_DIR}/val", Config.VAL_MASK_DIR, augment=False)
    if len(train_ds) == 0:
        print("❌ No data! Check paths.")
        return
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                          num_workers=Config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    print(f"\n📦 Train: {len(train_ds)} | Val: {len(val_ds)}")
    trainer = SETRTrainer(Config)
    if Config.RESUME_TRAINING:
        trainer.load_checkpoint()
    trainer.run(train_dl, val_dl)

if __name__ == "__main__":
    main()
