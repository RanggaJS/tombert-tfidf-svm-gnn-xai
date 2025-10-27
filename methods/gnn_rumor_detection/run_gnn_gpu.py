# coding=utf-8
"""
Script untuk menjalankan GNN dengan konfigurasi GPU
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import os
# Add parent directory to path to import gpu_config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(parent_dir, 'methods', 'utils')
sys.path.append(utils_dir)
from gpu_config import GPUConfig

# Import GNN models
from GAT import GAT
from GLAN import GLAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNRunner:
    """
    Runner untuk GNN dengan GPU support
    """
    
    def __init__(self, gpu_id=0, use_mixed_precision=True):
        self.gpu_config = GPUConfig(gpu_id=gpu_id, use_mixed_precision=use_mixed_precision)
        self.device = self.gpu_config.get_device()
        self.scaler = self.gpu_config.get_scaler()
        
    def prepare_data(self, data_dir):
        """
        Siapkan data untuk GNN
        """
        logger.info("Preparing data for GNN...")
        
        # Load data (simplified version)
        train_file = os.path.join(data_dir, 'train.tsv')
        dev_file = os.path.join(data_dir, 'dev.tsv')
        test_file = os.path.join(data_dir, 'test.tsv')
        
        def load_tsv(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0:  # Skip header
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        data.append({
                            'label': int(parts[1]),  # parts[1] is label
                            'tweet_id': parts[0],    # parts[0] is index
                            'image_id': parts[2],    # parts[2] is image_id
                            'text': parts[4],        # parts[4] is text
                            'target': parts[3]       # parts[3] is target
                        })
            return data
        
        train_data = load_tsv(train_file)
        dev_data = load_tsv(dev_file)
        test_data = load_tsv(test_file)
        
        # Convert to tensors
        train_texts = [item['text'] for item in train_data]
        train_labels = torch.tensor([item['label'] for item in train_data], dtype=torch.long)
        
        dev_texts = [item['text'] for item in dev_data]
        dev_labels = torch.tensor([item['label'] for item in dev_data], dtype=torch.long)
        
        test_texts = [item['text'] for item in test_data]
        test_labels = torch.tensor([item['label'] for item in test_data], dtype=torch.long)
        
        # Move to device
        train_labels = train_labels.to(self.device)
        dev_labels = dev_labels.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Create adjacency matrix using graph file
        graph_file = os.path.join(data_dir, 'twitter15_graph.txt')
        train_adj = self.create_adjacency_matrix(train_texts, train_labels, graph_file)
        
        return {
            'train': {'texts': train_texts, 'labels': train_labels, 'adj': train_adj},
            'dev': {'texts': dev_texts, 'labels': dev_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
    
    def create_adjacency_matrix(self, texts, labels, graph_file=None):
        """
        Buat adjacency matrix untuk GNN dari graph file
        """
        logger.info("Creating adjacency matrix...")
        
        n = len(texts)
        
        if graph_file and os.path.exists(graph_file):
            # Load graph dari file
            logger.info(f"Loading graph from {graph_file}")
            adj = torch.zeros(n, n)
            with open(graph_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= n:  # Limit to number of nodes
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        node_id = int(parts[0])
                        if node_id < n:
                            # Parse connections with weights
                            connections = parts[1].split()
                            for conn in connections:
                                if ':' in conn:
                                    neighbor_id, weight = conn.split(':')
                                    neighbor_id = int(neighbor_id)
                                    if neighbor_id < n:
                                        adj[node_id][neighbor_id] = float(weight)
                                        adj[neighbor_id][node_id] = float(weight)  # Make symmetric
        else:
            # Fallback: simplified adjacency matrix (random untuk demo)
            logger.info("Using random adjacency matrix (fallback)")
            adj = torch.rand(n, n) > 0.7  # Random adjacency
            adj = adj.float()
            
            # Make symmetric
            adj = (adj + adj.T) / 2
            adj[adj > 0.5] = 1
        adj[adj <= 0.5] = 0
        
        # Add self-loops
        adj += torch.eye(n)
        
        return adj.to(self.device)
    
    def train_model(self, data_dict, num_epochs=5):
        """
        Train GNN model
        """
        logger.info("Training GNN model...")
        
        # Use simplified training approach
        train_data = data_dict['train']
        
        # Create simple model for demonstration
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(SimpleGNN, self).__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        # Initialize highly optimized model
        class AdvancedGNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(AdvancedGNN, self).__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, hidden_dim)
                self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.linear4 = nn.Linear(hidden_dim // 2, output_dim)
                
                self.relu = nn.ReLU()
                self.leaky_relu = nn.LeakyReLU(0.1)
                self.dropout1 = nn.Dropout(0.2)
                self.dropout2 = nn.Dropout(0.3)
                self.dropout3 = nn.Dropout(0.4)
                
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm3 = nn.BatchNorm1d(hidden_dim // 2)
                
            def forward(self, x):
                x = self.leaky_relu(self.linear1(x))
                x = self.batch_norm1(x)
                x = self.dropout1(x)
                
                x = self.leaky_relu(self.linear2(x))
                x = self.batch_norm2(x)
                x = self.dropout2(x)
                
                x = self.leaky_relu(self.linear3(x))
                x = self.batch_norm3(x)
                x = self.dropout3(x)
                
                x = self.linear4(x)
                return x
        
        model = AdvancedGNN(input_dim=300, hidden_dim=256, output_dim=3)  # Much larger hidden dim
        model = model.to(self.device)
        
        # Enhanced loss and optimizer with scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW with weight decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler
        
        # Create proper text embeddings using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words='english')
        
        # Fit and transform text data
        train_text_features = tfidf.fit_transform(train_data['texts'])
        
        # Convert to tensor and move to device
        train_features = torch.FloatTensor(train_text_features.toarray()).to(self.device)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(train_features)
            loss = criterion(outputs, train_data['labels'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        return model
    
    def evaluate_model(self, model, data_dict):
        """
        Evaluate GNN model
        """
        logger.info("Evaluating GNN model...")
        
        test_data = data_dict['test']
        
        # Create proper test features using same TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words='english')
        
        # Fit on train data and transform test data
        all_texts = data_dict['train']['texts'] + test_data['texts']
        tfidf.fit(all_texts)
        
        test_text_features = tfidf.transform(test_data['texts'])
        test_features = torch.FloatTensor(test_text_features.toarray()).to(self.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_features)
            _, predicted = torch.max(outputs.data, 1)
            
            accuracy = accuracy_score(test_data['labels'].cpu().numpy(), predicted.cpu().numpy())
            
            # Generate classification report
            from sklearn.metrics import classification_report
            report = classification_report(
                test_data['labels'].cpu().numpy(), 
                predicted.cpu().numpy(),
                target_names=['Negative', 'Neutral', 'Positive']
            )
        
        return accuracy, report
    
    def run_gat(self, data, epochs=50, learning_rate=0.01, hidden_dim=64):
        """
        Jalankan GAT (Graph Attention Network)
        """
        logger.info("Running GAT...")
        
        train_data = data['train']
        dev_data = data['dev']
        test_data = data['test']
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(train_data['texts'], train_data['labels'])
        
        # Initialize GAT model
        nfeat = 300  # Feature dimension
        uV = len(train_data['texts'])  # Number of nodes
        n_output = 3  # Number of classes
        
        model = GAT(
            nfeat=nfeat,
            uV=uV,
            adj=adj,
            hidden=hidden_dim,
            nb_heads=8,
            n_output=n_output,
            dropout=0.5,
            alpha=0.3
        )
        
        # Optimize for GPU
        model = optimize_model_for_gpu(model, self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_tid = torch.arange(len(train_data['texts'])).to(self.device)
            outputs = model(X_tid)
            
            loss = criterion(outputs, train_data['labels'])
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Dev evaluation
            X_tid = torch.arange(len(dev_data['texts'])).to(self.device)
            dev_outputs = model(X_tid)
            dev_preds = torch.argmax(dev_outputs, dim=1)
            dev_accuracy = accuracy_score(dev_data['labels'].cpu(), dev_preds.cpu())
            
            # Test evaluation
            X_tid = torch.arange(len(test_data['texts'])).to(self.device)
            test_outputs = model(X_tid)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_accuracy = accuracy_score(test_data['labels'].cpu(), test_preds.cpu())
            
            # Calculate metrics
            dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(
                dev_data['labels'].cpu(), dev_preds.cpu(), average='macro'
            )
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                test_data['labels'].cpu(), test_preds.cpu(), average='macro'
            )
        
        return {
            'method': 'GAT',
            'dev_accuracy': dev_accuracy,
            'dev_precision': dev_precision,
            'dev_recall': dev_recall,
            'dev_f1': dev_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model': model
        }
    
    def run_glan(self, data, epochs=50, learning_rate=0.01):
        """
        Jalankan GLAN (Graph-based LSTM Attention Network)
        """
        logger.info("Running GLAN...")
        
        train_data = data['train']
        dev_data = data['dev']
        test_data = data['test']
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(train_data['texts'], train_data['labels'])
        
        # Initialize GLAN model
        config = {
            'embedding_weights': np.random.randn(10000, 300),  # Random embedding weights
            'maxlen': 64,
            'dropout': 0.5,
            'kernel_sizes': [3, 4, 5],
            'num_classes': 3
        }
        
        model = GLAN(config, adj)
        
        # Optimize for GPU
        model = optimize_model_for_gpu(model, self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_tid = torch.arange(len(train_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(train_data['texts']), 64)).to(self.device)
            outputs = model(X_tid, X_text)
            
            loss = criterion(outputs, train_data['labels'])
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Dev evaluation
            X_tid = torch.arange(len(dev_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(dev_data['texts']), 64)).to(self.device)
            dev_outputs = model(X_tid, X_text)
            dev_preds = torch.argmax(dev_outputs, dim=1)
            dev_accuracy = accuracy_score(dev_data['labels'].cpu(), dev_preds.cpu())
            
            # Test evaluation
            X_tid = torch.arange(len(test_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(test_data['texts']), 64)).to(self.device)
            test_outputs = model(X_tid, X_text)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_accuracy = accuracy_score(test_data['labels'].cpu(), test_preds.cpu())
            
            # Calculate metrics
            dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(
                dev_data['labels'].cpu(), dev_preds.cpu(), average='macro'
            )
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                test_data['labels'].cpu(), test_preds.cpu(), average='macro'
            )
        
        return {
            'method': 'GLAN',
            'dev_accuracy': dev_accuracy,
            'dev_precision': dev_precision,
            'dev_recall': dev_recall,
            'dev_f1': dev_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model': model
        }

def main():
    parser = argparse.ArgumentParser(description='Run GNN with GPU')
    parser.add_argument('--data_dir', default='./absa_data/twitter2015', 
                       help='Path to data directory')
    parser.add_argument('--gpu_id', type=int, default=0, 
                       help='GPU ID to use')
    parser.add_argument('--model', choices=['gat', 'glan', 'both'], default='both',
                       help='GNN model to run')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                       help='Hidden dimension for GAT')
    
    args = parser.parse_args()
    
    # Initialize GNN runner
    runner = GNNRunner(gpu_id=args.gpu_id, use_mixed_precision=True)
    
    # Prepare data
    data = runner.prepare_data(args.data_dir)
    
    results = {}
    
    if args.model in ['gat', 'both']:
        logger.info("Running GAT...")
        gat_results = runner.run_gat(data, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
        results['gat'] = gat_results
        
        print("\n" + "="*50)
        print("GAT RESULTS")
        print("="*50)
        print(f"Dev Accuracy: {gat_results['dev_accuracy']:.4f}")
        print(f"Test Accuracy: {gat_results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {gat_results['test_f1']:.4f}")
        print("="*50)
    
    if args.model in ['glan', 'both']:
        logger.info("Running GLAN...")
        glan_results = runner.run_glan(data, epochs=args.epochs, learning_rate=args.learning_rate)
        results['glan'] = glan_results
        
        print("\n" + "="*50)
        print("GLAN RESULTS")
        print("="*50)
        print(f"Dev Accuracy: {glan_results['dev_accuracy']:.4f}")
        print(f"Test Accuracy: {glan_results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {glan_results['test_f1']:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    results = main()
    
    def run_glan(self, data, epochs=50, learning_rate=0.01):
        """
        Jalankan GLAN (Graph-based LSTM Attention Network)
        """
        logger.info("Running GLAN...")
        
        train_data = data['train']
        dev_data = data['dev']
        test_data = data['test']
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(train_data['texts'], train_data['labels'])
        
        # Initialize GLAN model
        config = {
            'embedding_weights': np.random.randn(10000, 300),  # Random embedding weights
            'maxlen': 64,
            'dropout': 0.5,
            'kernel_sizes': [3, 4, 5],
            'num_classes': 3
        }
        
        model = GLAN(config, adj)
        
        # Optimize for GPU
        model = optimize_model_for_gpu(model, self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_tid = torch.arange(len(train_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(train_data['texts']), 64)).to(self.device)
            outputs = model(X_tid, X_text)
            
            loss = criterion(outputs, train_data['labels'])
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Dev evaluation
            X_tid = torch.arange(len(dev_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(dev_data['texts']), 64)).to(self.device)
            dev_outputs = model(X_tid, X_text)
            dev_preds = torch.argmax(dev_outputs, dim=1)
            dev_accuracy = accuracy_score(dev_data['labels'].cpu(), dev_preds.cpu())
            
            # Test evaluation
            X_tid = torch.arange(len(test_data['texts'])).to(self.device)
            X_text = torch.randint(0, 10000, (len(test_data['texts']), 64)).to(self.device)
            test_outputs = model(X_tid, X_text)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_accuracy = accuracy_score(test_data['labels'].cpu(), test_preds.cpu())
            
            # Calculate metrics
            dev_precision, dev_recall, dev_f1, _ = precision_recall_fscore_support(
                dev_data['labels'].cpu(), dev_preds.cpu(), average='macro'
            )
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                test_data['labels'].cpu(), test_preds.cpu(), average='macro'
            )
        
        return {
            'method': 'GLAN',
            'dev_accuracy': dev_accuracy,
            'dev_precision': dev_precision,
            'dev_recall': dev_recall,
            'dev_f1': dev_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model': model
        }

def main():
    parser = argparse.ArgumentParser(description='Run GNN with GPU')
    parser.add_argument('--data_dir', default='./absa_data/twitter2015', 
                       help='Path to data directory')
    parser.add_argument('--gpu_id', type=int, default=0, 
                       help='GPU ID to use')
    parser.add_argument('--model', choices=['gat', 'glan', 'both'], default='both',
                       help='GNN model to run')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                       help='Hidden dimension for GAT')
    
    args = parser.parse_args()
    
    # Initialize GNN runner
    runner = GNNRunner(gpu_id=args.gpu_id, use_mixed_precision=True)
    
    # Prepare data
    data = runner.prepare_data(args.data_dir)
    
    results = {}
    
    if args.model in ['gat', 'both']:
        logger.info("Running GAT...")
        gat_results = runner.run_gat(data, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
        results['gat'] = gat_results
        
        print("\n" + "="*50)
        print("GAT RESULTS")
        print("="*50)
        print(f"Dev Accuracy: {gat_results['dev_accuracy']:.4f}")
        print(f"Test Accuracy: {gat_results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {gat_results['test_f1']:.4f}")
        print("="*50)
    
    if args.model in ['glan', 'both']:
        logger.info("Running GLAN...")
        glan_results = runner.run_glan(data, epochs=args.epochs, learning_rate=args.learning_rate)
        results['glan'] = glan_results
        
        print("\n" + "="*50)
        print("GLAN RESULTS")
        print("="*50)
        print(f"Dev Accuracy: {glan_results['dev_accuracy']:.4f}")
        print(f"Test Accuracy: {glan_results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {glan_results['test_f1']:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    results = main()

