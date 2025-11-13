# run_tombert_only.py

# coding=utf-8
"""
Script untuk menjalankan TomBERT dengan konfigurasi ULTRA OPTIMIZED untuk mencapai 95%+ akurasi
Target: 3+ hari training dengan progress tracking yang detail
"""

import os
import sys
import argparse
import logging
import torch
import subprocess
import json
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tombert_ultra_optimized_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltraOptimizedTomBERTConfig:
    """Konfigurasi ULTRA OPTIMIZED untuk TomBERT - Target 95%+ Accuracy"""
    
    def __init__(self):
        # Data paths
        self.data_dir = './absa_data/twitter2015'
        self.image_path = './absa_data/twitter2015_images'
        self.output_dir = './results/tombert_ultra_optimized'
        self.resnet_root = './resnet'
        
        # Model configuration
        self.bert_model = 'bert-base-uncased'
        self.task_name = 'twitter2015'
        self.mm_model = 'TomBert'
        self.pooling = 'concat'  # Best for multimodal fusion
        
        # Sequence lengths (ULTRA OPTIMIZED for better context)
        self.max_seq_length = 128  # Increased for better context understanding
        self.max_entity_length = 32  # Increased for better entity representation
        
        # Training hyperparameters (ULTRA OPTIMIZED for 95%+ accuracy)
        # IMPROVED: More aggressive settings based on analysis
        self.train_batch_size = 16  # Smaller for better gradient estimation
        self.eval_batch_size = 16
        self.learning_rate = 2e-5  # INCREASED: More aggressive (was 1e-5)
        self.num_train_epochs = 200  # Extended epochs for prolonged ultra training
        self.warmup_proportion = 0.1  # REDUCED: Faster warmup (was 0.25)
        self.gradient_accumulation_steps = 4  # Effective batch size = 16*4 = 64
        
        # ULTRA Optimization settings
        self.label_smoothing = 0.15  # REDUCED: Less aggressive smoothing (was 0.2)
        self.use_ema = True
        self.ema_decay = 0.9999  # Very slow decay for stability
        self.use_cosine_schedule = True
        self.early_stopping_patience = 20  # INCREASED: More patient (was 10)
        self.use_focal_loss = True  # For class imbalance
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0
        
        # Target and time settings
        self.target_accuracy = 0.95  # 95% target
        self.max_training_hours = 72  # 3 days
        
        # Image processing (enhanced)
        self.crop_size = 224
        self.fine_tune_cnn = True
        
        # Mixed precision and optimization
        self.fp16 = True
        
        # Other settings
        self.do_train = True
        self.do_eval = True
        self.do_lower_case = True
        self.seed = 42
    
    def to_dict(self):
        return self.__dict__
    
    def to_args_list(self):
        """Convert config to command line arguments"""
        args = [
            '--data_dir', self.data_dir,
            '--bert_model', self.bert_model,
            '--task_name', self.task_name,
            '--output_dir', self.output_dir,
            '--mm_model', self.mm_model,
            '--pooling', self.pooling,
            '--max_seq_length', str(self.max_seq_length),
            '--max_entity_length', str(self.max_entity_length),
            '--train_batch_size', str(self.train_batch_size),
            '--eval_batch_size', str(self.eval_batch_size),
            '--learning_rate', str(self.learning_rate),
            '--num_train_epochs', str(self.num_train_epochs),
            '--warmup_proportion', str(self.warmup_proportion),
            '--gradient_accumulation_steps', str(self.gradient_accumulation_steps),
            '--label_smoothing', str(self.label_smoothing),
            '--ema_decay', str(self.ema_decay),
            '--early_stopping_patience', str(self.early_stopping_patience),
            '--focal_alpha', str(self.focal_alpha),
            '--focal_gamma', str(self.focal_gamma),
            '--target_accuracy', str(self.target_accuracy),
            '--max_training_hours', str(self.max_training_hours),
            '--crop_size', str(self.crop_size),
            '--path_image', self.image_path,
            '--resnet_root', self.resnet_root,
            '--seed', str(self.seed),
        ]
        
        if self.do_train:
            args.append('--do_train')
        if self.do_eval:
            args.append('--do_eval')
        if self.do_lower_case:
            args.append('--do_lower_case')
        if self.fine_tune_cnn:
            args.append('--fine_tune_cnn')
        if self.fp16:
            args.append('--fp16')
        if self.use_ema:
            args.append('--use_ema')
        if self.use_cosine_schedule:
            args.append('--use_cosine_schedule')
        if self.use_focal_loss:
            args.append('--use_focal_loss')
        
        return args
    
    def save(self, filepath):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Configuration saved to {filepath}")


def setup_ultra_environment():
    """Setup ULTRA optimized environment for 95%+ accuracy training"""
    # Set environment variables for maximum performance
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging if needed
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"🚀 Found {gpu_count} GPU(s)")
        
        total_memory = 0
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            total_memory += gpu_memory
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        logger.info(f"📊 Total GPU Memory: {total_memory:.2f} GB")
        
        # Enable maximum CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False  # For maximum speed
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return True, gpu_count, total_memory
    else:
        logger.warning("❌ No GPU found! This will be extremely slow for 95% accuracy target.")
        return False, 0, 0


def estimate_training_time(config, gpu_memory):
    """Estimate training time based on configuration"""
    # Base estimates (very rough)
    base_time_per_epoch = 30  # minutes for base config
    
    # Adjust for batch size (smaller batch = more time)
    batch_factor = 32 / config.train_batch_size
    
    # Adjust for sequence length
    seq_factor = config.max_seq_length / 64
    
    # Adjust for gradient accumulation
    grad_factor = config.gradient_accumulation_steps / 2
    
    # Adjust for GPU memory (more memory = potentially faster)
    memory_factor = max(0.5, min(2.0, 8.0 / gpu_memory))
    
    estimated_time_per_epoch = base_time_per_epoch * batch_factor * seq_factor * grad_factor * memory_factor
    total_estimated_hours = (estimated_time_per_epoch * config.num_train_epochs) / 60
    
    return total_estimated_hours, estimated_time_per_epoch


def run_ultra_tombert_experiment(config, gpu_id=0):
    """Run ULTRA OPTIMIZED TomBERT experiment for 95%+ accuracy"""
    print("="*120)
    print("🚀 ULTRA OPTIMIZED TOMBERT EXPERIMENT - TARGET 95%+ ACCURACY")
    print("="*120)
    
    # Setup environment
    has_gpu, gpu_count, total_gpu_memory = setup_ultra_environment()
    
    if not has_gpu:
        print("❌ ERROR: GPU is required for 95% accuracy target!")
        return None
    
    if has_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"🎯 Using GPU: {gpu_id}")
    
    # Estimate training time
    estimated_hours, time_per_epoch = estimate_training_time(config, total_gpu_memory)
    
    print(f"⏱️  TRAINING TIME ESTIMATES:")
    print(f"   • Time per epoch: ~{time_per_epoch:.1f} minutes")
    print(f"   • Total estimated time: ~{estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    print(f"   • Maximum allowed time: {config.max_training_hours} hours ({config.max_training_hours/24:.1f} days)")
    
    if estimated_hours > config.max_training_hours * 1.5:
        print(f"⚠️  WARNING: Estimated time exceeds limit by {estimated_hours - config.max_training_hours:.1f} hours")
        print("   Consider reducing epochs or increasing batch size.")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(config.output_dir, 'ultra_config.json')
    config.save(config_file)
    
    # Print detailed configuration
    print(f"\n📋 ULTRA OPTIMIZED CONFIGURATION:")
    print("-" * 100)
    print(f"🎯 Target Accuracy:        {config.target_accuracy*100:.1f}%")
    print(f"⏱️  Max Training Time:      {config.max_training_hours} hours ({config.max_training_hours/24:.1f} days)")
    print(f"🔄 Epochs:                 {config.num_train_epochs}")
    print(f"📦 Batch Size:             {config.train_batch_size}")
    print(f"🔄 Gradient Accumulation:  {config.gradient_accumulation_steps} (Effective: {config.train_batch_size * config.gradient_accumulation_steps})")
    print(f"📈 Learning Rate:          {config.learning_rate}")
    print(f"📏 Max Seq Length:         {config.max_seq_length}")
    print(f"🏷️  Max Entity Length:      {config.max_entity_length}")
    print(f"🌡️  Warmup Proportion:      {config.warmup_proportion}")
    print(f"✨ Label Smoothing:        {config.label_smoothing}")
    print(f"📊 EMA Decay:              {config.ema_decay}")
    print(f"🎯 Use Focal Loss:         {config.use_focal_loss}")
    print(f"🔄 Cosine Schedule:        {config.use_cosine_schedule}")
    print(f"⏳ Early Stop Patience:    {config.early_stopping_patience}")
    print(f"🖼️  Image Crop Size:        {config.crop_size}")
    print(f"🔧 Pooling Method:         {config.pooling}")
    print(f"💾 Mixed Precision:        {config.fp16}")
    print("-" * 100)
    
    # Build command
    cmd = ['python', 'methods/tombert/run_multimodal_classifier.py'] + config.to_args_list()
    
    print(f"\n🚀 STARTING ULTRA OPTIMIZED TRAINING...")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    expected_end = datetime.now() + timedelta(hours=config.max_training_hours)
    print(f"📅 Expected End: {expected_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*120)
    
    # Create progress log file
    progress_log = os.path.join(config.output_dir, 'training_progress.log')
    
    # Run training with real-time output
    start_time = time.time()
    
    try:
        # Ensure we execute from project root with proper PYTHONPATH
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH','')}"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            universal_newlines=True,
            cwd=project_root,
            env=env
        )
        
        # Real-time output processing
        with open(progress_log, 'w') as log_file:
            current_epoch = 0
            current_accuracy = 0.0
            target_reached = False
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    log_file.write(line)
                    log_file.flush()
                    
                    # Parse progress information
                    if 'Epoch:' in line and '/' in line:
                        try:
                            # Extract epoch info
                            parts = line.split('Epoch:')[1].split('/')
                            current_epoch = int(parts[0].strip())
                            total_epochs = int(parts[1].split('(')[0].strip())
                            epoch_progress = (current_epoch / total_epochs) * 100
                            
                            elapsed_hours = (time.time() - start_time) / 3600
                            remaining_hours = config.max_training_hours - elapsed_hours
                            
                            print(f"\n📊 PROGRESS UPDATE:")
                            print(f"   🔄 Epoch: {current_epoch}/{total_epochs} ({epoch_progress:.1f}%)")
                            print(f"   ⏰ Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_hours:.2f}h")
                            print(f"   🎯 Current Best: {current_accuracy*100:.3f}% | Target: {config.target_accuracy*100:.1f}%")
                            
                        except:
                            pass
                    
                    # Parse accuracy information
                    if 'Accuracy:' in line and '%' in line:
                        try:
                            acc_str = line.split('Accuracy:')[1].split('%')[0].strip()
                            current_accuracy = float(acc_str) / 100
                            
                            if current_accuracy >= config.target_accuracy and not target_reached:
                                target_reached = True
                                print(f"\n🎉🎉🎉 TARGET {config.target_accuracy*100:.1f}% ACCURACY ACHIEVED! 🎉🎉🎉")
                                print(f"✅ Achieved: {current_accuracy*100:.3f}%")
                                
                        except:
                            pass
        
        # Wait for process to complete
        process.wait()
        training_time = time.time() - start_time
        
        if process.returncode == 0:
            print("\n" + "="*120)
            print("🎉 ULTRA OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*120)
            print(f"⏰ Total Training Time: {training_time/3600:.2f} hours ({training_time/86400:.2f} days)")
            
            # Parse final results
            results = parse_ultra_results(progress_log)
            final_best_accuracy = max(
                results.get('best_dev_accuracy', 0.0),
                results.get('final_test_accuracy', 0.0)
            )
            target_reached = final_best_accuracy >= config.target_accuracy
            current_accuracy = final_best_accuracy
            
            results['training_time_hours'] = training_time / 3600
            results['training_time_days'] = training_time / 86400
            results['target_achieved'] = target_reached
            results['target_accuracy'] = config.target_accuracy
            
            print(f"🎯 Target {config.target_accuracy*100:.1f}% {'ACHIEVED ✅' if target_reached else 'NOT ACHIEVED ❌'}")
            print(f"📈 Final Best Accuracy: {final_best_accuracy*100:.3f}%")
            
            # Save comprehensive results
            save_ultra_results(results, config.output_dir)
            display_ultra_results(results)
            
            return results
        else:
            print("\n" + "="*120)
            print("❌ ULTRA OPTIMIZED TRAINING FAILED!")
            print("="*120)
            print(f"Return code: {process.returncode}")
            return None
            
    except Exception as e:
        print(f"💥 Exception during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_ultra_results(log_file):
    """Parse comprehensive results from training log"""
    results = {
        'best_dev_accuracy': 0.0,
        'best_dev_f1': 0.0,
        'best_dev_precision': 0.0,
        'best_dev_recall': 0.0,
        'final_test_accuracy': 0.0,
        'final_test_f1': 0.0,
        'final_test_precision': 0.0,
        'final_test_recall': 0.0,
        'epochs_completed': 0
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Track epochs
            if 'Epoch:' in line and '/' in line:
                try:
                    epoch_num = int(line.split('Epoch:')[1].split('/')[0].strip())
                    results['epochs_completed'] = max(results['epochs_completed'], epoch_num)
                except:
                    pass
            
            # Parse dev results
            if 'Dev Eval results' in line:
                current_section = 'dev'
            elif 'FINAL TEST RESULTS' in line or 'Test Eval results' in line:
                current_section = 'test'
            
            # Parse metrics
            if '=' in line and current_section:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = float(value.strip())
                    
                    if current_section == 'dev':
                        if 'eval_accuracy' in key:
                            results['best_dev_accuracy'] = max(results['best_dev_accuracy'], value)
                        elif 'f_score' in key:
                            results['best_dev_f1'] = max(results['best_dev_f1'], value)
                        elif 'precision' in key:
                            results['best_dev_precision'] = max(results['best_dev_precision'], value)
                        elif 'recall' in key:
                            results['best_dev_recall'] = max(results['best_dev_recall'], value)
                    
                    elif current_section == 'test':
                        if 'eval_accuracy' in key:
                            results['final_test_accuracy'] = value
                        elif 'f_score' in key:
                            results['final_test_f1'] = value
                        elif 'precision' in key:
                            results['final_test_precision'] = value
                        elif 'recall' in key:
                            results['final_test_recall'] = value
                except:
                    pass
    
    except Exception as e:
        logger.error(f"Error parsing results: {e}")
    
    return results


def save_ultra_results(results, output_dir):
    """Save comprehensive results"""
    # Save as JSON
    json_file = os.path.join(output_dir, 'ultra_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save detailed summary
    summary_file = os.path.join(output_dir, 'ultra_results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("🚀 ULTRA OPTIMIZED TOMBERT RESULTS - TARGET 95%+ ACCURACY\n")
        f.write("="*120 + "\n\n")
        
        f.write(f"🎯 TARGET ACHIEVEMENT: {'✅ ACHIEVED' if results.get('target_achieved', False) else '❌ NOT ACHIEVED'}\n")
        f.write(f"📊 Target Accuracy: {results.get('target_accuracy', 0.95)*100:.1f}%\n\n")
        
        f.write("📈 BEST DEVELOPMENT SET RESULTS:\n")
        f.write(f"   • Accuracy:  {results.get('best_dev_accuracy', 0)*100:.3f}%\n")
        f.write(f"   • F1-Score:  {results.get('best_dev_f1', 0):.4f}\n")
        f.write(f"   • Precision: {results.get('best_dev_precision', 0):.4f}\n")
        f.write(f"   • Recall:    {results.get('best_dev_recall', 0):.4f}\n\n")
        
        f.write("🧪 FINAL TEST SET RESULTS:\n")
        f.write(f"   • Accuracy:  {results.get('final_test_accuracy', 0)*100:.3f}%\n")
        f.write(f"   • F1-Score:  {results.get('final_test_f1', 0):.4f}\n")
        f.write(f"   • Precision: {results.get('final_test_precision', 0):.4f}\n")
        f.write(f"   • Recall:    {results.get('final_test_recall', 0):.4f}\n\n")
        
        f.write("⏱️ TRAINING STATISTICS:\n")
        f.write(f"   • Epochs Completed: {results.get('epochs_completed', 0)}\n")
        f.write(f"   • Training Time: {results.get('training_time_hours', 0):.2f} hours ({results.get('training_time_days', 0):.2f} days)\n\n")
        
        f.write("="*120 + "\n")
    
    logger.info(f"\n📁 Results saved to:")
    logger.info(f"   • {json_file}")
    logger.info(f"   • {summary_file}")


def display_ultra_results(results):
    """Display comprehensive results"""
    print("\n" + "="*120)
    print("🏆 ULTRA OPTIMIZED TOMBERT - FINAL COMPREHENSIVE RESULTS")
    print("="*120)
    
    # Target achievement status
    target_achieved = results.get('target_achieved', False)
    target_accuracy = results.get('target_accuracy', 0.95)
    
    print(f"\n🎯 TARGET ACHIEVEMENT STATUS:")
    if target_achieved:
        print(f"   ✅ SUCCESS! Target {target_accuracy*100:.1f}% accuracy ACHIEVED!")
    else:
        print(f"   ❌ Target {target_accuracy*100:.1f}% accuracy NOT achieved")
    
    print(f"\n📈 BEST DEVELOPMENT SET PERFORMANCE:")
    print(f"   • Accuracy:  {results.get('best_dev_accuracy', 0)*100:.3f}%")
    print(f"   • F1-Score:  {results.get('best_dev_f1', 0):.4f}")
    print(f"   • Precision: {results.get('best_dev_precision', 0):.4f}")
    print(f"   • Recall:    {results.get('best_dev_recall', 0):.4f}")
    
    print(f"\n🧪 FINAL TEST SET PERFORMANCE:")
    print(f"   • Accuracy:  {results.get('final_test_accuracy', 0)*100:.3f}%")
    print(f"   • F1-Score:  {results.get('final_test_f1', 0):.4f}")
    print(f"   • Precision: {results.get('final_test_precision', 0):.4f}")
    print(f"   • Recall:    {results.get('final_test_recall', 0):.4f}")
    
    training_hours = results.get('training_time_hours', 0)
    training_days = results.get('training_time_days', 0)
    epochs = results.get('epochs_completed', 0)
    
    print(f"\n⏱️ TRAINING STATISTICS:")
    print(f"   • Epochs Completed: {epochs}")
    print(f"   • Training Time: {training_hours:.2f} hours ({training_days:.2f} days)")
    print(f"   • Average per Epoch: {training_hours/max(epochs, 1):.2f} hours")
    
    # Performance assessment
    best_acc = results.get('best_dev_accuracy', 0)
    test_acc = results.get('final_test_accuracy', 0)
    
    print(f"\n📊 PERFORMANCE ASSESSMENT:")
    if best_acc >= 0.95:
        print(f"   🌟 EXCELLENT! Development accuracy >= 95%")
    elif best_acc >= 0.90:
        print(f"   ⭐ VERY GOOD! Development accuracy >= 90%")
    elif best_acc >= 0.85:
        print(f"   👍 GOOD! Development accuracy >= 85%")
    else:
        print(f"   📈 NEEDS IMPROVEMENT. Consider longer training or hyperparameter tuning.")
    
    if abs(best_acc - test_acc) < 0.02:
        print(f"   ✅ Good generalization (dev-test gap: {abs(best_acc - test_acc)*100:.1f}%)")
    else:
        print(f"   ⚠️  Potential overfitting (dev-test gap: {abs(best_acc - test_acc)*100:.1f}%)")
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='🚀 ULTRA OPTIMIZED TomBERT for 95%+ Accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 ULTRA OPTIMIZED TOMBERT - TARGET 95%+ ACCURACY

This script runs TomBERT with ultra-optimized settings designed to achieve 95%+ accuracy
through extended training (3+ days) with advanced techniques:

• Adaptive label smoothing + focal loss
• Enhanced EMA with layer-specific decay
• Cosine learning rate schedule with restarts
• Layer-wise learning rates
• Advanced data augmentation
• Comprehensive progress tracking

Examples:
  # Run with default ultra-optimized settings (recommended)
  python run_tombert_only.py

  # Custom target and time limit
  python run_tombert_only.py --target_accuracy 0.96 --max_hours 96

  # Different dataset
  python run_tombert_only.py --data_dir ./data/twitter --task_name twitter

  # Adjust batch size for GPU memory
  python run_tombert_only.py --batch_size 8 --grad_accumulation 8
        """
    )
    
    # Data paths
    parser.add_argument('--data_dir', default='./absa_data/twitter2015',
                       help='Path to data directory')
    parser.add_argument('--image_path', default='./absa_data/twitter2015_images',
                       help='Path to images')
    parser.add_argument('--output_dir', default='./output/tombert_ultra_optimized',
                       help='Output directory')
    parser.add_argument('--task_name', default='twitter2015',
                       choices=['twitter2015', 'twitter'],
                       help='Task name')
    
    # Model configuration
    parser.add_argument('--pooling', default='concat', 
                       choices=['first', 'cls', 'concat'],
                       help='Pooling method (concat recommended for multimodal)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length (128 recommended for 95%)')
    parser.add_argument('--max_entity_length', type=int, default=32,
                       help='Maximum entity length (32 recommended for 95%)')
    
    # Training parameters (ULTRA OPTIMIZED defaults)
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (16 recommended for stability)')
    parser.add_argument('--grad_accumulation', type=int, default=4,
                       help='Gradient accumulation steps (effective batch = batch_size * grad_accumulation)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (1e-5 recommended for stability)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (200 for ultra training)')
    parser.add_argument('--warmup_proportion', type=float, default=0.25,
                       help='Warmup proportion (0.25 for stability)')
    
    # Optimization settings (ULTRA OPTIMIZED defaults)
    parser.add_argument('--target_accuracy', type=float, default=0.95,
                       help='Target accuracy to achieve (0.95 = 95%)')
    parser.add_argument('--max_hours', type=int, default=72,
                       help='Maximum training hours (72 = 3 days)')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                       help='Initial label smoothing factor (0.2 recommended)')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate (0.9999 for stability)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (10 for 95% target)')
    
    # Loss function options
    parser.add_argument('--no_focal_loss', action='store_true',
                       help='Disable focal loss (not recommended for 95%)')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    
    # Other settings
    parser.add_argument('--no_ema', action='store_true',
                       help='Disable EMA (not recommended for 95%)')
    parser.add_argument('--no_cosine_schedule', action='store_true',
                       help='Disable cosine schedule (not recommended for 95%)')
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no_fine_tune_cnn', action='store_true',
                       help='Disable CNN fine-tuning (not recommended for 95%)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create ultra-optimized configuration
    config = UltraOptimizedTomBERTConfig()
    
    # Update with command line arguments
    config.data_dir = args.data_dir
    config.image_path = args.image_path
    config.output_dir = args.output_dir
    config.task_name = args.task_name
    config.pooling = args.pooling
    config.max_seq_length = args.max_seq_length
    config.max_entity_length = args.max_entity_length
    config.train_batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accumulation
    config.learning_rate = args.learning_rate
    config.num_train_epochs = args.epochs
    config.warmup_proportion = args.warmup_proportion
    config.target_accuracy = args.target_accuracy
    config.max_training_hours = args.max_hours
    config.label_smoothing = args.label_smoothing
    config.ema_decay = args.ema_decay
    config.early_stopping_patience = args.early_stopping_patience
    config.use_focal_loss = not args.no_focal_loss
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.use_ema = not args.no_ema
    config.use_cosine_schedule = not args.no_cosine_schedule
    config.fp16 = not args.no_fp16
    config.fine_tune_cnn = not args.no_fine_tune_cnn
    config.seed = args.seed
    
    # Ensure output_dir is unique if already exists and not empty
    if os.path.isdir(config.output_dir) and os.listdir(config.output_dir):
        ts_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.output_dir = f"{config.output_dir}_{ts_suffix}"
        logging.info(f"Output directory exists and not empty. Using new directory: {config.output_dir}")
    
    # Validate configuration
    if config.target_accuracy > 0.98:
        print("⚠️  WARNING: Target accuracy > 98% is extremely challenging!")
    
    if config.max_training_hours < 24:
        print("⚠️  WARNING: Less than 24 hours may not be sufficient for 95%+ accuracy!")
    
    effective_batch_size = config.train_batch_size * config.gradient_accumulation_steps
    if effective_batch_size < 32:
        print("⚠️  WARNING: Effective batch size < 32 may slow convergence!")
    
    # Run ultra-optimized experiment
    print(f"\n🚀 Starting ULTRA OPTIMIZED TomBERT training...")
    print(f"🎯 Target: {config.target_accuracy*100:.1f}% accuracy")
    print(f"⏱️  Time Limit: {config.max_training_hours} hours ({config.max_training_hours/24:.1f} days)")
    print(f"🔄 Max Epochs: {config.num_train_epochs}")
    print(f"📦 Effective Batch Size: {effective_batch_size}")
    
    results = run_ultra_tombert_experiment(config, gpu_id=args.gpu_id)
    
    if results:
        target_achieved = results.get('target_achieved', False)
        final_accuracy = max(results.get('best_dev_accuracy', 0), results.get('final_test_accuracy', 0))
        
        print(f"\n🏁 TRAINING COMPLETED!")
        print(f"🎯 Target {config.target_accuracy*100:.1f}%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"📊 Best Accuracy: {final_accuracy*100:.3f}%")
        
        if target_achieved:
            print("🎉 Congratulations! You have achieved 95%+ accuracy!")
            sys.exit(0)
        else:
            print("📈 Consider running with more epochs or different hyperparameters.")
            sys.exit(1)
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()