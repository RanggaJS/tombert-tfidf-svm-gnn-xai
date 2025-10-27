# coding=utf-8
"""
Optimized script untuk menjalankan eksperimen TF-IDF + SVM
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tfidf_svm_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_results_to_file(results, output_dir):
    """Save detailed results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'method': results['method'],
            'status': results['status'],
            'execution_time': results['execution_time'],
            'dev_accuracy': float(results['dev_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'timestamp': datetime.now().isoformat()
        }, f, indent=4)
    
    # Save reports
    with open(os.path.join(output_dir, 'dev_report.txt'), 'w') as f:
        f.write(results['dev_report'])
    
    with open(os.path.join(output_dir, 'test_report.txt'), 'w') as f:
        f.write(results['test_report'])
    
    logger.info(f"Results saved to {output_dir}")

def run_tfidf_svm_experiment():
    """Run optimized TF-IDF + SVM experiment"""
    logger.info("="*60)
    logger.info("RUNNING OPTIMIZED TF-IDF + SVM EXPERIMENT")
    logger.info("="*60)
    
    try:
        # Import optimized classifier
        sys.path.append('./methods/tfidf_svm')
        from classical_methods import TFIDFSVMClassifier, load_absa_data, prepare_image_paths
        
        start_time = time.time()
        
        # Configuration
        data_dir = './absa_data/twitter2015'
        image_base_path = './absa_data/twitter2015_images'
        output_dir = './results/tfidf_svm_results'
        
        # Load data
        logger.info("Loading data...")
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
        
        # Initialize optimized classifier
        logger.info("Initializing optimized classifier...")
        classifier = TFIDFSVMClassifier(
            max_features=30000,
            ngram_range=(1, 3),
            use_grid_search=False,  # Set True for best results (slower)
            random_state=42
        )
        
        # Train model
        logger.info("Training model...")
        classifier.fit(train_texts, train_image_paths, train_labels)
        
        # Evaluate on development set
        logger.info("Evaluating on development set...")
        dev_accuracy, dev_report = classifier.evaluate(dev_texts, dev_image_paths, dev_labels)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_accuracy, test_report = classifier.evaluate(test_texts, test_image_paths, test_labels)
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'optimized_tfidf_svm_model.pkl')
        classifier.save_model(model_path)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare results
        result = {
            'method': 'Optimized TF-IDF + SVM',
            'status': 'completed',
            'execution_time': execution_time,
            'dev_accuracy': dev_accuracy,
            'test_accuracy': test_accuracy,
            'dev_report': dev_report,
            'test_report': test_report,
            'output_dir': output_dir
        }
        
        # Save results
        save_results_to_file(result, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZED TF-IDF + SVM EXPERIMENT RESULTS")
        print("="*60)
        print(f"Method: {result['method']}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"\nDevelopment Set:")
        print(f"  Accuracy: {dev_accuracy:.4f}")
        print(f"\nTest Set:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"\nDetailed Reports:")
        print("\nDevelopment Set Report:")
        print(dev_report)
        print("\nTest Set Report:")
        print(test_report)
        print("="*60)
        print(f"\n✅ Results saved to: {output_dir}")
        
        return result
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        return {
            'method': 'Optimized TF-IDF + SVM',
            'status': 'failed',
            'error': str(e)
        }

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("STARTING OPTIMIZED TF-IDF + SVM EXPERIMENT")
    logger.info("="*60)
    
    result = run_tfidf_svm_experiment()
    
    if result['status'] == 'completed':
        print(f"\n✅ Experiment completed successfully!")
    else:
        print(f"\n❌ Experiment failed: {result.get('error', 'Unknown error')}")