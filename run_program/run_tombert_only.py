# coding=utf-8
"""
Script untuk menjalankan TomBERT dengan konfigurasi optimal - OPTIMIZED VERSION
"""

import os
import sys
import argparse
import logging
import torch
import subprocess
import json
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tombert_optimized_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedTomBERTConfig:
    """Konfigurasi optimal untuk TomBERT"""
    
    def __init__(self):
        # Data paths
        self.data_dir = './absa_data/twitter2015'
        self.image_path = './absa_data/twitter2015_images'
        self.output_dir = './results/tombert_optimized'
        self.resnet_root = './resnet'
        
        # Model configuration
        self.bert_model = 'bert-base-uncased'
        self.task_name = 'twitter2015'
        self.mm_model = 'TomBert'
        self.pooling = 'first'  # Standard pooling for consistency
        
        # Sequence lengths (OPTIMIZED)
        self.max_seq_length = 64  # Standard length for consistency
        self.max_entity_length = 16  # Standard length for consistency
        
        # Training hyperparameters (OPTIMIZED for 90%+ accuracy)
        self.train_batch_size = 32  # Increased for better batch normalization
        self.eval_batch_size = 32
        self.learning_rate = 2e-5  # Optimized learning rate for faster convergence
        self.num_train_epochs = 20  # Increased for better convergence
        self.warmup_proportion = 0.2  # Extended warmup for stability
        self.gradient_accumulation_steps = 1  # No accumulation needed with larger batch
        
        # Optimization settings (OPTIMIZED)
        self.label_smoothing = 0.05  # Reduced for better discrimination
        self.use_ema = True
        self.ema_decay = 0.9995  # Slower decay for better stability
        self.use_cosine_schedule = True
        self.early_stopping_patience = 5  # Increased patience
        self.weight_decay = 1e-4  # L2 regularization
        self.dropout_rate = 0.1  # Dropout for regularization
        
        # Image processing
        self.crop_size = 224
        self.fine_tune_cnn = True
        
        # Mixed precision
        self.fp16 = True
        
        # Other
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
        
        return args
    
    def save(self, filepath):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Configuration saved to {filepath}")


def setup_environment():
    """Setup optimal environment for training"""
    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        return True
    else:
        logger.warning("No GPU found! Training will be slow.")
        return False


def run_tombert_experiment(config, gpu_id=0):
    """Run TomBERT experiment with given configuration"""
    logger.info("="*80)
    logger.info("RUNNING OPTIMIZED TOMBERT EXPERIMENT")
    logger.info("="*80)
    
    # Setup environment
    has_gpu = setup_environment()
    
    if has_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Using GPU: {gpu_id}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(config.output_dir, 'config.json')
    config.save(config_file)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info("-" * 80)
    for key, value in config.to_dict().items():
        logger.info(f"  {key:30s}: {value}")
    logger.info("-" * 80)
    
    # Build command
    cmd = ['python', 'run_multimodal_classifier.py'] + config.to_args_list()
    
    logger.info("\nCommand:")
    logger.info(" ".join(cmd))
    logger.info("\n" + "="*80)
    
    # Run training
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info("\n" + "="*80)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            
            # Parse and display results
            results = parse_results(result.stdout)
            results['training_time'] = training_time
            
            # Save results
            save_results(results, config.output_dir)
            
            # Display results
            display_results(results)
            
            return results
        else:
            logger.error("\n" + "="*80)
            logger.error("❌ TRAINING FAILED!")
            logger.error("="*80)
            logger.error("Error output:")
            logger.error(result.stderr)
            return None
            
    except Exception as e:
        logger.error(f"Exception during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_results(output):
    """Parse training output to extract metrics"""
    results = {
        'dev_accuracy': 0.0,
        'dev_precision': 0.0,
        'dev_recall': 0.0,
        'dev_f1': 0.0,
        'test_accuracy': 0.0,
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1': 0.0
    }
    
    lines = output.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Detect section
        if 'Dev Eval results' in line:
            current_section = 'dev'
        elif 'Test Eval results' in line:
            current_section = 'test'
        
        # Parse metrics
        if '=' in line and current_section:
            try:
                key, value = line.split('=')
                key = key.strip()
                value = float(value.strip())
                
                if 'eval_accuracy' in key:
                    results[f'{current_section}_accuracy'] = value
                elif 'precision' in key:
                    results[f'{current_section}_precision'] = value
                elif 'recall' in key:
                    results[f'{current_section}_recall'] = value
                elif 'f_score' in key:
                    results[f'{current_section}_f1'] = value
            except:
                pass
    
    return results


def save_results(results, output_dir):
    """Save results to files"""
    # Save as JSON
    json_file = os.path.join(output_dir, 'results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save as text
    text_file = os.path.join(output_dir, 'results_summary.txt')
    with open(text_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OPTIMIZED TOMBERT RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Development Set:\n")
        f.write(f"  Accuracy:  {results.get('dev_accuracy', 0):.4f}\n")
        f.write(f"  Precision: {results.get('dev_precision', 0):.4f}\n")
        f.write(f"  Recall:    {results.get('dev_recall', 0):.4f}\n")
        f.write(f"  F1-Score:  {results.get('dev_f1', 0):.4f}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  Accuracy:  {results.get('test_accuracy', 0):.4f}\n")
        f.write(f"  Precision: {results.get('test_precision', 0):.4f}\n")
        f.write(f"  Recall:    {results.get('test_recall', 0):.4f}\n")
        f.write(f"  F1-Score:  {results.get('test_f1', 0):.4f}\n\n")
        
        f.write(f"Training Time: {results.get('training_time', 0):.2f} seconds\n")
        f.write("="*80 + "\n")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {json_file}")
    logger.info(f"  - {text_file}")


def display_results(results):
    """Display results in a nice format"""
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print("\nDevelopment Set:")
    print(f"  Accuracy:  {results.get('dev_accuracy', 0):.4f}")
    print(f"  Precision: {results.get('dev_precision', 0):.4f}")
    print(f"  Recall:    {results.get('dev_recall', 0):.4f}")
    print(f"  F1-Score:  {results.get('dev_f1', 0):.4f}")
    
    print("\nTest Set:")
    print(f"  Accuracy:  {results.get('test_accuracy', 0):.4f}")
    print(f"  Precision: {results.get('test_precision', 0):.4f}")
    print(f"  Recall:    {results.get('test_recall', 0):.4f}")
    print(f"  F1-Score:  {results.get('test_f1', 0):.4f}")
    
    training_time = results.get('training_time', 0)
    print(f"\nTraining Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Optimized TomBERT Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_tombert_only.py

  # Run with custom settings
  python run_tombert_only.py --output_dir ./my_output --epochs 15 --batch_size 16

  # Run without EMA
  python run_tombert_only.py --no_ema

  # Use different pooling method
  python run_tombert_only.py --pooling first
        """
    )
    
    # Data paths
    parser.add_argument('--data_dir', default='./absa_data/twitter2015',
                       help='Path to data directory')
    parser.add_argument('--image_path', default='./absa_data/twitter2015_images',
                       help='Path to images')
    parser.add_argument('--output_dir', default='./output/tombert_optimized',
                       help='Output directory')
    
    # Model configuration
    parser.add_argument('--pooling', default='concat', 
                       choices=['first', 'cls', 'concat'],
                       help='Pooling method (concat recommended)')
    parser.add_argument('--max_seq_length', type=int, default=80,
                       help='Maximum sequence length')
    parser.add_argument('--max_entity_length', type=int, default=20,
                       help='Maximum entity length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--warmup_proportion', type=float, default=0.15,
                       help='Warmup proportion')
    
    # Optimization settings
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--no_ema', action='store_true',
                       help='Disable exponential moving average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                       help='EMA decay rate')
    parser.add_argument('--no_cosine_schedule', action='store_true',
                       help='Disable cosine learning rate schedule')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    
    # Other settings
    parser.add_argument('--no_fp16', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--no_fine_tune_cnn', action='store_true',
                       help='Disable CNN fine-tuning')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizedTomBERTConfig()
    
    # Update with command line arguments
    config.data_dir = args.data_dir
    config.image_path = args.image_path
    config.output_dir = args.output_dir
    config.pooling = args.pooling
    config.max_seq_length = args.max_seq_length
    config.max_entity_length = args.max_entity_length
    config.train_batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_train_epochs = args.epochs
    config.warmup_proportion = args.warmup_proportion
    config.label_smoothing = args.label_smoothing
    config.use_ema = not args.no_ema
    config.ema_decay = args.ema_decay
    config.use_cosine_schedule = not args.no_cosine_schedule
    config.early_stopping_patience = args.early_stopping_patience
    config.fp16 = not args.no_fp16
    config.fine_tune_cnn = not args.no_fine_tune_cnn
    config.seed = args.seed
    
    # Run experiment
    results = run_tombert_experiment(config, gpu_id=args.gpu_id)
    
    if results:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()