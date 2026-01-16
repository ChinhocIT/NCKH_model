"""
Train Adaptor + MaskedFeatureExtractor + VisionTextDecoder on VizWiz dataset.

Trains:
    1. MaskedFeatureExtractor - Extracts masked region features
    2. ImprovedVisionLanguageAdaptor - Converts vision→LLM tokens
    3. VisionTextDecoder - Generates text from vision tokens

Usage:
    python scripts/train_adaptor_vizwiz_v2.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# GDA components
from src.models.adaptor import ImprovedVisionLanguageAdaptor
from src.models.vit_encoder import MaskedFeatureExtractor
from src.models.text_decoder import VisionTextDecoder, VisionTextDecoderLoss

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, BitsAndBytesConfig

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
plt.style.use('seaborn-v0_8-whitegrid')  # Clean scientific style


class TrainingPlotter:
    """
    Scientific plotting class for training metrics.
    Generates publication-quality figures.
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # History tracking
        self.train_losses = []  # Per-batch losses
        self.train_epoch_losses = []  # Average per epoch
        self.val_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.epoch_times = []
        self.epochs = []
        
        print(f"📊 TrainingPlotter initialized - plots will be saved to {save_dir}")
    
    def update_batch(self, loss: float, lr: float = None, grad_norm: float = None):
        """Record batch-level metrics."""
        self.train_losses.append(loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
    
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, epoch_time: float, lr: float = None):
        """Record epoch-level metrics."""
        self.epochs.append(epoch)
        self.train_epoch_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch_times.append(epoch_time)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_loss_curves(self):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Per-batch training loss (smoothed)
        ax1 = axes[0]
        if len(self.train_losses) > 0:
            # Raw loss
            ax1.plot(self.train_losses, alpha=0.3, color='blue', label='Raw loss')
            
            # Smoothed loss (moving average)
            window = min(100, len(self.train_losses) // 10 + 1)
            if window > 1:
                smoothed = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.train_losses)), smoothed, 
                        color='blue', linewidth=2, label=f'Smoothed (window={window})')
            
            ax1.set_xlabel('Batch', fontsize=12)
            ax1.set_ylabel('Training Loss', fontsize=12)
            ax1.set_title('Training Loss per Batch', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Right: Epoch-level train vs val loss
        ax2 = axes[1]
        if len(self.epochs) > 0:
            ax2.plot(self.epochs, self.train_epoch_losses, 'b-o', linewidth=2, 
                    markersize=8, label='Train Loss')
            ax2.plot(self.epochs, self.val_losses, 'r-s', linewidth=2, 
                    markersize=8, label='Validation Loss')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # Add best val loss annotation
            if self.val_losses:
                best_idx = np.argmin(self.val_losses)
                best_val = self.val_losses[best_idx]
                ax2.annotate(f'Best: {best_val:.4f}', 
                            xy=(self.epochs[best_idx], best_val),
                            xytext=(self.epochs[best_idx] + 0.5, best_val + 0.5),
                            fontsize=10, color='red',
                            arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        save_path = self.save_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved loss curves: {save_path}")
    
    def plot_learning_rate(self):
        """Plot learning rate schedule."""
        if len(self.learning_rates) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.learning_rates, 'g-', linewidth=1.5)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'learning_rate.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved learning rate plot: {save_path}")
    
    def plot_gradient_norms(self):
        """Plot gradient norms over training."""
        if len(self.gradient_norms) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.gradient_norms, 'purple', alpha=0.5, linewidth=0.5)
        
        # Smoothed
        window = min(100, len(self.gradient_norms) // 10 + 1)
        if window > 1:
            smoothed = np.convolve(self.gradient_norms, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.gradient_norms)), smoothed, 
                   'purple', linewidth=2, label=f'Smoothed')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'gradient_norms.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved gradient norms: {save_path}")
    
    def plot_epoch_times(self):
        """Plot time per epoch."""
        if len(self.epoch_times) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(self.epochs, [t/60 for t in self.epoch_times], color='skyblue', edgecolor='navy')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (minutes)', fontsize=12)
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, t in zip(bars, self.epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{t/60:.1f}m', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / 'epoch_times.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved epoch times: {save_path}")
    
    def plot_summary(self):
        """Generate comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 12))
        
        # 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.epochs) > 0:
            ax1.plot(self.epochs, self.train_epoch_losses, 'b-o', linewidth=2, label='Train')
            ax1.plot(self.epochs, self.val_losses, 'r-s', linewidth=2, label='Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Batch loss distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.train_losses) > 0:
            ax2.hist(self.train_losses, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            ax2.axvline(np.mean(self.train_losses), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(self.train_losses):.2f}')
            ax2.axvline(np.median(self.train_losses), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(self.train_losses):.2f}')
            ax2.set_xlabel('Loss')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Loss Distribution', fontweight='bold')
            ax2.legend()
        
        # 3. Training progress (rolling mean)
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.train_losses) > 100:
            window = 100
            rolling = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            ax3.fill_between(range(len(rolling)), rolling, alpha=0.3, color='blue')
            ax3.plot(rolling, 'b-', linewidth=1)
            ax3.set_xlabel('Batch')
            ax3.set_ylabel('Loss (100-batch rolling mean)')
            ax3.set_title('Training Progress', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        if len(self.epochs) > 0:
            stats_text = [
                f"Training Summary",
                f"─" * 30,
                f"Total Epochs: {len(self.epochs)}",
                f"Total Batches: {len(self.train_losses):,}",
                f"",
                f"Final Train Loss: {self.train_epoch_losses[-1]:.4f}",
                f"Final Val Loss: {self.val_losses[-1]:.4f}",
                f"Best Val Loss: {min(self.val_losses):.4f} (Epoch {np.argmin(self.val_losses)+1})",
                f"",
                f"Avg Epoch Time: {np.mean(self.epoch_times)/60:.1f} min",
                f"Total Time: {sum(self.epoch_times)/3600:.2f} hours",
            ]
            ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('VizWiz Adaptor Training Report', fontsize=16, fontweight='bold', y=0.98)
        
        save_path = self.save_dir / 'training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved training summary: {save_path}")
    
    def save_history(self):
        """Save training history to JSON for later analysis."""
        history = {
            'epochs': self.epochs,
            'train_epoch_losses': self.train_epoch_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'train_batch_losses': self.train_losses[-1000:],  # Last 1000 batches
            'learning_rates': self.learning_rates[-100:] if self.learning_rates else [],
        }
        
        save_path = self.save_dir / 'training_history.json'
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Saved training history: {save_path}")
    
    def generate_all_plots(self):
        """Generate all plots at once."""
        print("\n📊 Generating all training plots...")
        self.plot_loss_curves()
        self.plot_learning_rate()
        self.plot_gradient_norms()
        self.plot_epoch_times()
        self.plot_summary()
        self.save_history()
        print("✅ All plots generated!\n")


class VizWizDataset(Dataset):
    """VizWiz dataset for training adaptor + text decoder."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_answer_length: int = 32
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_answer_length = max_answer_length
        self.samples = []
        
        # Find annotations
        possible_ann_paths = [
            self.data_dir / 'Annotations' / f'{split}.json',
            self.data_dir / 'Annotations' / 'Annotations' / f'{split}.json',
            self.data_dir / f'{split}.json',
        ]
        
        ann_file = None
        for path in possible_ann_paths:
            if path.exists():
                ann_file = path
                break
        
        if ann_file is None:
            raise FileNotFoundError(f"Cannot find {split}.json in {self.data_dir}")
        
        print(f"📂 Found annotations: {ann_file}")
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Find image directory
        possible_img_dirs = [
            self.data_dir / split / split,  # nested: val/val/
            self.data_dir / split,
            self.data_dir / 'images' / split,
        ]
        
        self.img_dir = None
        for path in possible_img_dirs:
            if path.exists() and path.is_dir():
                self.img_dir = path
                break
        
        if self.img_dir is None:
            raise FileNotFoundError(f"Cannot find image directory for {split}")
        
        print(f"📂 Found images: {self.img_dir}")
        
        # Build samples list
        skipped = 0
        for item in self.annotations:
            image_name = item.get('image')
            if not image_name:
                skipped += 1
                continue
            
            image_path = self.img_dir / image_name
            if not image_path.exists():
                skipped += 1
                continue
            
            question = item.get('question', '')
            answers = item.get('answers', [])
            
            if not answers:
                skipped += 1
                continue
            
            # Get majority answer
            answer_counts = {}
            for ans in answers:
                ans_text = ans.get('answer', '')
                if ans_text:
                    answer_counts[ans_text] = answer_counts.get(ans_text, 0) + 1
            
            if not answer_counts:
                skipped += 1
                continue
            
            best_answer = max(answer_counts, key=answer_counts.get)
            
            self.samples.append({
                'image_path': str(image_path),
                'question': question,
                'answer': best_answer
            })
        
        print(f"✅ Loaded {len(self.samples)} samples (skipped {skipped})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image_np = np.array(image)
        
        return {
            'image': image_np,
            'question': sample['question'],
            'answer': sample['answer']
        }


def collate_fn(batch):
    """Simple collate - just return as list."""
    return {
        'images': [item['image'] for item in batch],
        'questions': [item['question'] for item in batch],
        'answers': [item['answer'] for item in batch]
    }


class VizWizTrainer:
    """Trainer for Adaptor + MaskedFeatureExtractor + TextDecoder."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        use_fp16: bool = True,
        learning_rate: float = 1e-4
    ):
        self.device = device
        self.use_fp16 = use_fp16
        
        print("="*70)
        print("🚀 VizWiz Training - Adaptor + TextDecoder")
        print("="*70)
        
        # 1. Load Qwen2-VL
        print("\n📦 [1/4] Loading Qwen2-VL...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        self.full_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Get ViT encoder
        self.vit_encoder = self.full_model.visual
        self.vit_encoder.eval()
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        
        # Get dimensions
        try:
            self.vision_dim = self.vit_encoder.config.hidden_size
        except:
            self.vision_dim = 1536
        
        try:
            self.llm_dim = self.full_model.language_model.config.hidden_size
        except:
            self.llm_dim = 1536
        
        print(f"✅ Model loaded - Vision dim: {self.vision_dim}, LLM dim: {self.llm_dim}")
        
        # 2. Create trainable modules
        print("\n🔧 [2/4] Creating trainable modules...")
        
        self.masked_extractor = MaskedFeatureExtractor(
            feature_dim=self.vision_dim
        ).to(device)
        
        self.adaptor = ImprovedVisionLanguageAdaptor(
            vision_dim=self.vision_dim,
            llm_dim=self.llm_dim,
            num_query_tokens=64
        ).to(device)
        
        # Load Qwen weights into adaptor
        self.adaptor.load_qwen_weights_enhanced(self.vit_encoder)
        
        self.text_decoder = VisionTextDecoder(
            vision_dim=self.llm_dim,  # adaptor outputs llm_dim
            hidden_dim=512,
            vocab_size=self.tokenizer.vocab_size,
            num_decoder_layers=4,
            num_heads=8,
            max_length=32,
            dropout=0.1
        ).to(device)
        
        self.text_decoder.set_tokenizer(self.tokenizer)
        
        print(f"✅ Created MaskedFeatureExtractor, Adaptor, TextDecoder")
        
        # 3. Loss and optimizer
        print("\n📊 [3/4] Setting up optimizer...")
        
        self.loss_fn = VisionTextDecoderLoss(
            pad_token_id=self.tokenizer.pad_token_id or 151643
        )
        
        trainable_params = list(self.masked_extractor.parameters()) + \
                          list(self.adaptor.parameters()) + \
                          list(self.text_decoder.parameters())
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        self.scaler = GradScaler(enabled=use_fp16)
        
        total_params = sum(p.numel() for p in trainable_params)
        print(f"✅ Total trainable params: {total_params:,}")
        
        print("\n" + "="*70)
    
    def extract_features(self, image_rgb: np.ndarray) -> Optional[torch.Tensor]:
        """Extract ViT features from image using processor."""
        try:
            pil_image = Image.fromarray(image_rgb)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Describe."}
                ]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            pixel_values = inputs['pixel_values']
            
            # Check if already features (2D or 3D tensor)
            if pixel_values.dim() == 2:
                return pixel_values.unsqueeze(0).float()
            elif pixel_values.dim() == 3:
                return pixel_values.float()
            
            # Need to extract from 4D tensor
            if pixel_values.dim() == 4:
                B, C, H, W = pixel_values.shape
                patch_size = getattr(self.vit_encoder.config, 'patch_size', 14)
                grid_h = H // patch_size
                grid_w = W // patch_size
                
                grid_thw = torch.tensor(
                    [[1, grid_h, grid_w]], 
                    device=self.device, 
                    dtype=torch.long
                )
                
                with torch.no_grad():
                    try:
                        features = self.vit_encoder(
                            pixel_values=pixel_values, 
                            grid_thw=grid_thw
                        )
                    except:
                        features = self.vit_encoder(pixel_values)
                    
                    if isinstance(features, tuple):
                        features = features[0]
                    elif hasattr(features, 'last_hidden_state'):
                        features = features.last_hidden_state
                
                return features.float() if features.dtype == torch.float16 else features
            
            return None
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return None
    
    def train_step(self, batch) -> Optional[float]:
        """Single training step."""
        self.masked_extractor.train()
        self.adaptor.train()
        self.text_decoder.train()
        
        images = batch['images']
        answers = batch['answers']
        
        total_loss = 0.0
        valid_samples = 0
        
        for image, answer in zip(images, answers):
            # Extract ViT features
            vit_features = self.extract_features(image)
            if vit_features is None:
                continue
            
            # Create dummy mask (full image)
            H_f = W_f = int(np.sqrt(vit_features.shape[1]))
            dummy_mask = np.ones((H_f * 14, W_f * 14), dtype=np.uint8)
            
            # Tokenize answer
            answer_encoding = self.tokenizer(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            answer_ids = answer_encoding['input_ids']
            
            with autocast(enabled=self.use_fp16):
                # Forward: features → masked → adaptor → decoder → loss
                masked_features = self.masked_extractor(vit_features, dummy_mask)
                vision_tokens = self.adaptor(masked_features)
                logits = self.text_decoder(vision_tokens, target_ids=answer_ids)
                loss = self.loss_fn(logits, answer_ids)
            
            # Backward
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            valid_samples += 1
        
        if valid_samples > 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            return total_loss / valid_samples
        
        return None
    
    @torch.no_grad()
    def validate(self, dataloader, max_samples: int = 100):
        """Validate on dataset."""
        self.masked_extractor.eval()
        self.adaptor.eval()
        self.text_decoder.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            if num_samples >= max_samples:
                break
            
            images = batch['images']
            answers = batch['answers']
            
            for image, answer in zip(images, answers):
                vit_features = self.extract_features(image)
                if vit_features is None:
                    continue
                
                H_f = W_f = int(np.sqrt(vit_features.shape[1]))
                dummy_mask = np.ones((H_f * 14, W_f * 14), dtype=np.uint8)
                
                answer_encoding = self.tokenizer(
                    answer, max_length=32, padding='max_length',
                    truncation=True, return_tensors='pt'
                ).to(self.device)
                answer_ids = answer_encoding['input_ids']
                
                with autocast(enabled=self.use_fp16):
                    masked_features = self.masked_extractor(vit_features, dummy_mask)
                    vision_tokens = self.adaptor(masked_features)
                    logits = self.text_decoder(vision_tokens, target_ids=answer_ids)
                    loss = self.loss_fn(logits, answer_ids)
                
                total_loss += loss.item()
                num_samples += 1
        
        return total_loss / max(num_samples, 1)
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'masked_extractor_state_dict': self.masked_extractor.state_dict(),
            'adaptor_state_dict': self.adaptor.state_dict(),
            'text_decoder_state_dict': self.text_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
        print(f"💾 Saved checkpoint: {path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_dir: str = "checkpoints",
        save_every: int = 1
    ):
        """Full training loop with scientific plotting."""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        # Initialize plotter
        plotter = TrainingPlotter(save_dir=save_dir)
        
        print(f"\n🚀 Starting training for {epochs} epochs")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                if loss is not None:
                    train_losses.append(loss)
                    plotter.update_batch(loss)  # Track batch loss
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            
            # Validation
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Update plotter with epoch metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            plotter.update_epoch(epoch, avg_train_loss, val_loss, epoch_time, current_lr)
            
            print(f"\n📊 Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Time={epoch_time:.1f}s")
            
            # Save checkpoint
            if epoch % save_every == 0 or val_loss < best_val_loss:
                checkpoint_path = os.path.join(save_dir, f'adaptor_epoch{epoch}.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(save_dir, 'adaptor_best.pth')
                    self.save_checkpoint(best_path, epoch, val_loss)
            
            # Generate interim plots every 2 epochs
            if epoch % 2 == 0:
                plotter.plot_loss_curves()
        
        # Generate all final plots
        plotter.generate_all_plots()
        
        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Adaptor + TextDecoder on VizWiz')
    
    parser.add_argument('--data-dir', type=str, default='D:/luu_tam/gda/VizWiz',
                        help='Path to VizWiz dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (small due to memory)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints/adaptor_vizwiz',
                        help='Checkpoint save directory')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = VizWizTrainer(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True,
        learning_rate=args.lr
    )
    
    # Create datasets
    train_dataset = VizWizDataset(
        data_dir=args.data_dir,
        split='train'
    )
    
    val_dataset = VizWizDataset(
        data_dir=args.data_dir,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
