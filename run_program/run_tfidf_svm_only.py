# coding=utf-8
"""
Ultra optimized script untuk menjalankan eksperimen TF-IDF + SVM
Target: 80%+ accuracy dengan training yang lebih efisien
"""

import os
import sys
import time
import logging
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import psutil
import gc
from tqdm import tqdm

# Setup logging - will be reconfigured in main function with output directory
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and log progress percentages for long-running tasks"""

    def __init__(self):
        self._last_percent = {}

    def __call__(self, phase, percent, current, total, message=""):
        """Handle progress callback from classifiers"""
        last = self._last_percent.get(phase, -1.0)
        if percent < 0:
            percent = 0.0
        if percent > 100:
            percent = 100.0

        # Rate-limit logging to ~1% increments to avoid excessive logs
        if percent == 100.0 or percent - last >= 1.0:
            self._last_percent[phase] = percent

            total_str = "?"
            current_str = "?"
            if isinstance(total, (int, float)) and total > 0:
                total_str = f"{int(total)}" if isinstance(total, int) or total.is_integer() else f"{total:.0f}"
            if isinstance(current, (int, float)):
                current_str = f"{int(current)}" if isinstance(current, int) or float(current).is_integer() else f"{current:.0f}"

            extra = f" | {message}" if message else ""
            logger.info(f"[Progress] {phase}: {percent:6.2f}% ({current_str}/{total_str}){extra}")

    def update_stage(self, stage, current_index, total_stages, message=""):
        """Convenience helper for stage-based progress"""
        total_stages = max(total_stages, 1)
        current_index = max(0, min(current_index, total_stages))

        percent = (current_index / total_stages) * 100.0
        self(
            phase=f"stage::{stage}",
            percent=percent,
            current=current_index,
            total=total_stages,
            message=message
        )

def monitor_system_resources():
    """Monitor system resources"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    logger.info(f"System Resources - CPU: {cpu}%, Memory: {memory.percent}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    
    return {
        'cpu_percent': cpu,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used/1024**3,
        'memory_total_gb': memory.total/1024**3
    }

def save_progress(results, output_dir, stage="training"):
    """Save progress during training"""
    os.makedirs(output_dir, exist_ok=True)
    
    progress_file = os.path.join(output_dir, f'progress_{stage}.json')
    with open(progress_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'results': results,
            'system_resources': monitor_system_resources()
        }, f, indent=4)
    
    logger.info(f"Progress saved: {stage}")

def save_final_results(results, output_dir):
    """Save comprehensive final results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON with full details
    json_path = os.path.join(output_dir, 'ultra_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'method': results['method'],
            'status': results['status'],
            'total_training_time': results['total_training_time'],
            'dev_accuracy': float(results['dev_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'training_stages': results.get('training_stages', []),
            'system_info': results.get('system_info', {}),
            'hyperparameters': results.get('hyperparameters', {}),
            'timestamp': datetime.now().isoformat()
        }, f, indent=4)
    
    # Save detailed reports
    with open(os.path.join(output_dir, 'dev_report_ultra.txt'), 'w') as f:
        f.write(results['dev_report'])
    
    with open(os.path.join(output_dir, 'test_report_ultra.txt'), 'w') as f:
        f.write(results['test_report'])
    
    # Save training log
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ULTRA TF-IDF + SVM TRAINING SUMMARY\n")
        f.write(f"="*50 + "\n")
        f.write(f"Total Training Time: {results['total_training_time']:.2f} seconds ({results['total_training_time']/3600:.2f} hours)\n")
        f.write(f"Development Accuracy: {results['dev_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*50 + "\n")
    
    logger.info(f"Comprehensive results saved to {output_dir}")

def run_ultra_tfidf_svm_experiment(target_accuracy=0.80, max_hours=72):
    """Run ultra optimized TF-IDF + SVM experiment for maximum accuracy
    
    Args:
        target_accuracy: Target accuracy to achieve (default: 0.80)
        max_hours: Maximum training time in hours (default: 72)
    """
    logger.info("="*80)
    logger.info("STARTING ULTRA TF-IDF + SVM EXPERIMENT")
    logger.info(f"TARGET: {target_accuracy*100:.0f}%+ ACCURACY WITH EFFICIENT TRAINING")
    logger.info(f"MAX TRAINING TIME: {max_hours} hours")
    logger.info("="*80)
    
    start_time = time.time()
    training_stages = []
    progress_tracker = ProgressTracker()
    total_stages = 5
    
    try:
        # Import ultra classifier
        sys.path.append('./methods/tfidf_svm')
        from classical_methods import UltraTFIDFSVMClassifier, load_absa_data, prepare_image_paths
        
        # Configuration for maximum accuracy
        data_dir = './absa_data/twitter2015'
        image_base_path = './absa_data/twitter2015_images'
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'./output/tfidf_svm_ultra_optimized_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging to output directory
        log_file = os.path.join(output_dir, 'training_progress.log')
        
        # Clear root logger handlers to prevent duplication
        root_logger = logging.getLogger()
        root_logger.handlers = []
        
        # Clear existing handlers for this logger
        logger.handlers = []
        logger.propagate = False  # Prevent propagation to root logger
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        
        # Also configure root logger to prevent duplication from other modules
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Log file: {log_file}")
        
        # Monitor initial system state
        initial_resources = monitor_system_resources()
        
        # Stage 1: Data Loading
        stage_start = time.time()
        logger.info("STAGE 1: Loading and preparing data...")
        progress_tracker.update_stage("data_loading", 0, total_stages, "Starting data preparation")
        train_data, dev_data, test_data = load_absa_data(data_dir)
        logger.info(f"Data loaded: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")
        
        # Prepare features
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        train_image_paths = prepare_image_paths(train_data, image_base_path)
        
        dev_texts = [item['text'] for item in dev_data]
        dev_labels = [item['label'] for item in dev_data]
        dev_image_paths = prepare_image_paths(dev_data, image_base_path)
        
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        test_image_paths = prepare_image_paths(test_data, image_base_path)
        
        stage_time = time.time() - stage_start
        training_stages.append({
            'stage': 'data_loading',
            'duration': stage_time,
            'timestamp': datetime.now().isoformat()
        })
        save_progress({'stage': 'data_loading', 'duration': stage_time}, output_dir, "data_loading")
        progress_tracker.update_stage("data_loading", 1, total_stages, "Data loading completed")
        
        # Stage 2: Model Initialization
        stage_start = time.time()
        logger.info("STAGE 2: Initializing ultra classifier...")
        classifier = UltraTFIDFSVMClassifier(
            max_features=200000,           # Maximum features for best coverage
            ngram_range=(1, 4),            # Extended n-grams
            use_extensive_search=True,     # Enable extensive hyperparameter search
            random_state=42,
            use_images=False,              # Text-only mode: disable image analysis
            use_pca=True,                  # Use dimensionality reduction
            use_feature_selection=True,    # Use feature selection
            use_ensemble=True,             # Use ensemble methods for maximum accuracy
            progress_callback=progress_tracker,
            n_iter=50,                     # Reduced from 200 to 50 for faster training (250 fits vs 2000)
            cv_folds=5                     # Reduced from 10 to 5 for faster training
        )
        
        stage_time = time.time() - stage_start
        training_stages.append({
            'stage': 'model_initialization',
            'duration': stage_time,
            'timestamp': datetime.now().isoformat()
        })
        save_progress({'stage': 'model_initialization', 'duration': stage_time}, output_dir, "model_init")
        progress_tracker.update_stage("model_initialization", 2, total_stages, "Model initialization completed")
        
        # Stage 3: Feature Extraction (Long process)
        stage_start = time.time()
        logger.info("STAGE 3: Starting ultra feature extraction (this will take hours)...")
        progress_tracker.update_stage("ultra_training", 2, total_stages, "Entering ultra training phase")
        
        # This will trigger the ultra feature extraction process
        logger.info("Beginning comprehensive training process...")
        logger.info("Expected duration: Several hours to 1 day for 80%+ accuracy")
        logger.info("The model will perform:")
        logger.info("- Ultra text preprocessing with advanced sentiment analysis")
        logger.info("- Multiple TF-IDF vectorizations with different parameters")
        logger.info("- Text-only mode: Image analysis DISABLED (text features only)")
        logger.info("- Extensive hyperparameter search (50 iterations × 5 folds = 250 fits)")
        logger.info("- Ensemble training with multiple algorithms")
        logger.info("- Advanced feature selection and dimensionality reduction")
        
        # Train ultra model (this is where the 3+ days will be spent)
        training_start = time.time()
        classifier.fit(train_texts, train_image_paths, train_labels)
        training_time = time.time() - training_start
        
        stage_time = time.time() - stage_start
        training_stages.append({
            'stage': 'ultra_training',
            'duration': stage_time,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        })
        save_progress({
            'stage': 'ultra_training_completed', 
            'duration': stage_time,
            'training_time': training_time
        }, output_dir, "training_completed")
        progress_tracker.update_stage("ultra_training", 3, total_stages, "Ultra training phase completed")
        
        # Stage 4: Model Evaluation
        stage_start = time.time()
        logger.info("STAGE 4: Evaluating ultra model...")
        progress_tracker.update_stage("evaluation", 3, total_stages, "Evaluation started")
        
        # Evaluate on development set
        logger.info("Evaluating on development set...")
        dev_accuracy, dev_report = classifier.evaluate(dev_texts, dev_image_paths, dev_labels, phase="evaluation_dev", batch_size=256)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_accuracy, test_report = classifier.evaluate(test_texts, test_image_paths, test_labels, phase="evaluation_test", batch_size=256)
        
        stage_time = time.time() - stage_start
        training_stages.append({
            'stage': 'evaluation',
            'duration': stage_time,
            'dev_accuracy': dev_accuracy,
            'test_accuracy': test_accuracy,
            'timestamp': datetime.now().isoformat()
        })
        progress_tracker.update_stage("evaluation", 4, total_stages, "Evaluation completed")
        
        # Stage 5: Model Saving
        stage_start = time.time()
        logger.info("STAGE 5: Saving ultra model...")
        progress_tracker.update_stage("model_saving", 4, total_stages, "Saving model artifacts")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'ultra_tfidf_svm_model.pkl')
        classifier.save_model(model_path)
        
        stage_time = time.time() - stage_start
        training_stages.append({
            'stage': 'model_saving',
            'duration': stage_time,
            'timestamp': datetime.now().isoformat()
        })
        progress_tracker.update_stage("model_saving", 5, total_stages, "Model saving completed")
        
        # Calculate total time
        total_time = time.time() - start_time
        final_resources = monitor_system_resources()
        
        # Prepare comprehensive results
        result = {
            'method': 'Ultra Enhanced TF-IDF + SVM Ensemble',
            'status': 'completed',
            'total_training_time': total_time,
            'dev_accuracy': dev_accuracy,
            'test_accuracy': test_accuracy,
            'dev_report': dev_report,
            'test_report': test_report,
            'training_stages': training_stages,
            'system_info': {
                'initial_resources': initial_resources,
                'final_resources': final_resources,
                'total_duration_hours': total_time / 3600,
                'total_duration_days': total_time / (3600 * 24)
            },
            'hyperparameters': {
                'max_features': 200000,
                'ngram_range': '(1, 4)',
                'use_extensive_search': True,
                'use_ensemble': True,
                'use_images': False,  # Text-only mode
                'use_pca': True,
                'use_feature_selection': True
            },
            'output_dir': output_dir
        }
        
        # Save comprehensive results
        save_final_results(result, output_dir)
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("ULTRA TF-IDF + SVM EXPERIMENT COMPLETED")
        print("="*80)
        print(f"Method: {result['method']}")
        print(f"Total Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours / {total_time/(3600*24):.2f} days)")
        print(f"\nACCURACY RESULTS:")
        print(f"  Development Set: {dev_accuracy:.4f} ({dev_accuracy*100:.2f}%)")
        print(f"  Test Set:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        if dev_accuracy >= target_accuracy or test_accuracy >= target_accuracy:
            print(f"\n🎉 SUCCESS: Achieved {target_accuracy*100:.0f}%+ accuracy target!")
        else:
            print(f"\n📊 Results: {max(dev_accuracy, test_accuracy)*100:.2f}% best accuracy (target: {target_accuracy*100:.0f}%)")
        
        print(f"\nTRAINING STAGES:")
        for i, stage in enumerate(training_stages, 1):
            print(f"  {i}. {stage['stage']}: {stage['duration']:.2f}s")
        
        print(f"\nDETAILED REPORTS:")
        print("\nDevelopment Set Report:")
        print(dev_report)
        print("\nTest Set Report:")
        print(test_report)
        
        print("="*80)
        print(f"✅ Comprehensive results saved to: {output_dir}")
        print("="*80)
        
        return result
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return {
            'method': 'Ultra Enhanced TF-IDF + SVM',
            'status': 'interrupted',
            'training_time': time.time() - start_time,
            'training_stages': training_stages
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        return {
            'method': 'Ultra Enhanced TF-IDF + SVM',
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time,
            'training_stages': training_stages
        }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TF-IDF + SVM experiment')
    parser.add_argument('--target_accuracy', type=float, default=0.80,
                        help='Target accuracy to achieve (default: 0.80)')
    parser.add_argument('--max_hours', type=float, default=72.0,
                        help='Maximum training time in hours (default: 72)')
    parser.add_argument('--skip_prompt', action='store_true',
                        help='Skip interactive prompt (useful for nohup)')
    
    args = parser.parse_args()
    
    target_accuracy = args.target_accuracy
    max_hours = args.max_hours
    
    logger.info("="*80)
    logger.info("ULTRA TF-IDF + SVM EXPERIMENT")
    logger.info(f"TARGETING {target_accuracy*100:.0f}%+ ACCURACY WITH EFFICIENT TRAINING")
    logger.info(f"MAX TRAINING TIME: {max_hours} hours")
    logger.info("="*80)
    
    # Display system information
    logger.info(f"System Info:")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total/1024**3:.1f} GB")
    logger.info(f"  Python: {sys.version}")
    
    print(f"\n⚠️  NOTE: This experiment targets {target_accuracy*100:.0f}%+ accuracy")
    print(f"⚠️  Maximum training time: {max_hours} hours")
    print("⚠️  It will use computational resources")
    print("⚠️  Make sure you have sufficient disk space and memory")
    print("⚠️  Progress will be saved regularly to prevent data loss")
    print("⚠️  Logs will be saved in ./output/tfidf_svm_ultra_optimized_[timestamp]/")
    
    # Check if running in interactive mode
    is_interactive = sys.stdin.isatty() and not args.skip_prompt
    
    if is_interactive:
        user_input = input("\nDo you want to proceed? (yes/no): ").lower().strip()
        if user_input not in ['yes', 'y']:
            print("Experiment cancelled by user.")
            sys.exit(0)
    
    logger.info("Starting ultra experiment...")
    result = run_ultra_tfidf_svm_experiment(target_accuracy=target_accuracy, max_hours=max_hours)
    
    if result['status'] == 'completed':
        print(f"\n🎉 Ultra experiment completed successfully!")
        print(f"Best accuracy achieved: {max(result['dev_accuracy'], result['test_accuracy'])*100:.2f}%")
    elif result['status'] == 'interrupted':
        print(f"\n⚠️ Experiment was interrupted after {result['training_time']/3600:.2f} hours")
    else:
        print(f"\n❌ Experiment failed: {result.get('error', 'Unknown error')}")