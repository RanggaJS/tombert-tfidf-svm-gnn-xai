# coding=utf-8
"""
Implementasi metode klasik TF-IDF + SVM untuk analisis sentimen multimodal (OPTIMIZED)
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
from PIL import Image
import pickle
import joblib
from tqdm import tqdm
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTextPreprocessor:
    """Advanced text preprocessing for better feature extraction"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
        
        # Sentiment lexicons
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                   'love', 'best', 'perfect', 'beautiful', 'awesome', 'brilliant'])
        self.negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                                   'poor', 'disappointing', 'useless', 'waste'])
    
    def preprocess(self, text):
        """Advanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle negations (important for sentiment)
        text = re.sub(r"n't", " not", text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_sentiment_features(self, text):
        """Extract sentiment-based features"""
        words = text.lower().split()
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Exclamation and question marks
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Text length features
        text_length = len(text)
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return np.array([pos_count, neg_count, exclamation_count, question_count, 
                        text_length, word_count, avg_word_length])

class EnhancedImageFeatureExtractor:
    """Enhanced image feature extraction"""
    
    def __init__(self):
        self.target_size = (128, 128)  # Increased for better features
    
    def extract_features(self, img_path):
        """Extract comprehensive image features"""
        try:
                img = cv2.imread(img_path)
                if img is None:
                return np.zeros(200)  # Increased feature size
                
            img = cv2.resize(img, self.target_size)
                
            # Multiple feature types
                color_features = self._extract_color_features(img)
                texture_features = self._extract_texture_features(img)
            edge_features = self._extract_edge_features(img)
            statistical_features = self._extract_statistical_features(img)
            
            # Combine all features
            all_features = np.concatenate([
                    color_features,
                    texture_features, 
                edge_features,
                statistical_features
            ])
            
            # Ensure fixed size
            if len(all_features) < 200:
                all_features = np.pad(all_features, (0, 200 - len(all_features)), 'constant')
            else:
                all_features = all_features[:200]
            
            return all_features
                
            except Exception as e:
            logger.warning(f"Error extracting features from {img_path}: {e}")
            return np.zeros(200)
    
    def _extract_color_features(self, img):
        """Enhanced color feature extraction"""
        # Multiple color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # HSV histograms (more bins for better representation)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
        
        # Color moments
        hsv_mean = np.mean(hsv.reshape(-1, 3), axis=0)
        hsv_std = np.std(hsv.reshape(-1, 3), axis=0)
        
        # Dominant colors (simplified)
        pixels = img.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        
        return np.concatenate([hist_h, hist_s, hist_v, hsv_mean, hsv_std, dominant_color])
    
    def _extract_texture_features(self, img):
        """Extract texture features using multiple methods"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (optimized)
        lbp = self._compute_lbp(gray)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-7)
        
        # Gabor filters (simplified)
        gabor_features = self._compute_gabor_features(gray)
        
        return np.concatenate([hist_lbp, gabor_features])
    
    def _compute_lbp(self, gray):
        """Compute Local Binary Pattern"""
        lbp = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i, j] = code
        return lbp
    
    def _compute_gabor_features(self, gray):
        """Compute Gabor filter features"""
        features = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((5, 5), 1.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
        return np.array(features)
    
    def _extract_edge_features(self, img):
        """Extract edge-based features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        return np.array([
            edge_density,
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude)
        ])
    
    def _extract_statistical_features(self, img):
        """Extract statistical features"""
        features = []
        for channel in cv2.split(img):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        return np.array(features)

class TFIDFSVMClassifier:
    """
    Enhanced TF-IDF + SVM classifier for multimodal sentiment analysis
    """
    
    def __init__(self, max_features=30000, ngram_range=(1, 3), random_state=42, 
                 use_grid_search=False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.use_grid_search = use_grid_search
        
        # Text preprocessor
        self.text_preprocessor = AdvancedTextPreprocessor()
        
        # Enhanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            token_pattern=r'\b\w+\b'
        )
        
        # Image feature extractor
        self.image_extractor = EnhancedImageFeatureExtractor()
        
        # Scalers
        self.text_scaler = StandardScaler()
        self.image_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        
        # PCA for dimensionality reduction (optional)
        self.use_pca = True
        self.pca_text = PCA(n_components=0.95, random_state=random_state)  # Keep 95% variance
        
        # Enhanced SVM classifier (OPTIMIZED for 90%+ accuracy)
        if use_grid_search:
            self.svm_classifier = None  # Will be set after grid search
        else:
            self.svm_classifier = SVC(
                kernel='rbf',
                C=100.0,  # Increased for better fit
                gamma='auto',  # Optimized gamma
                random_state=random_state,
                probability=True,
                class_weight='balanced',
                cache_size=3000,  # Increased cache
                decision_function_shape='ovo',  # One-vs-one for better accuracy
                shrinking=True,
                tol=1e-3  # Optimized tolerance
            )
        
        self.is_fitted = False
    
    def _preprocess_texts(self, texts):
        """Preprocess all texts"""
        return [self.text_preprocessor.preprocess(text) for text in texts]
    
    def _extract_sentiment_features(self, texts):
        """Extract sentiment features from texts"""
        features = []
        for text in texts:
            features.append(self.text_preprocessor.extract_sentiment_features(text))
        return np.array(features)
    
    def extract_text_features(self, texts, fit=True):
        """Extract enhanced text features"""
        logger.info("Extracting text features...")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # TF-IDF features
        if fit:
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        
        # Sentiment features
        sentiment_features = self._extract_sentiment_features(processed_texts)
        
        return tfidf_features, sentiment_features
    
    def extract_image_features(self, image_paths):
        """Extract image features"""
        logger.info(f"Extracting features from {len(image_paths)} images...")
        features = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            features.append(self.image_extractor.extract_features(img_path))
        
        return np.array(features)
    
    def _perform_grid_search(self, X_train, y_train):
        """Perform grid search for best SVM parameters"""
        logger.info("Performing grid search for optimal parameters...")
        
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(random_state=self.random_state, probability=True, 
                  class_weight='balanced', cache_size=2000)
        
        grid_search = GridSearchCV(
            svm, param_grid, cv=3, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def fit(self, texts, image_paths, labels):
        """Train the model"""
        logger.info("Training enhanced TF-IDF + SVM model...")
        
        # Extract text features
        tfidf_features, sentiment_features = self.extract_text_features(texts, fit=True)
        
        # Extract image features
        image_features = self.extract_image_features(image_paths)
        
        # Convert sparse to dense
        tfidf_dense = tfidf_features.toarray()
        
        # Apply PCA to text features if enabled
        if self.use_pca:
            logger.info("Applying PCA to text features...")
            tfidf_dense = self.pca_text.fit_transform(tfidf_dense)
            logger.info(f"Text features reduced to {tfidf_dense.shape[1]} dimensions")
        
        # Scale features
        tfidf_scaled = self.text_scaler.fit_transform(tfidf_dense)
        image_scaled = self.image_scaler.fit_transform(image_features)
        sentiment_scaled = self.sentiment_scaler.fit_transform(sentiment_features)
        
        # OPTIMIZED: Enhanced weighted fusion for better accuracy
        combined_features = np.hstack([
            tfidf_scaled * 2.0,      # Text features (increased weight for better accuracy)
            sentiment_scaled * 1.5,   # Sentiment features (increased weight)
            image_scaled * 1.2        # Image features (slightly increased weight)
        ])
        
        # Train SVM
        if self.use_grid_search:
            self.svm_classifier = self._perform_grid_search(combined_features, labels)
        else:
        self.svm_classifier.fit(combined_features, labels)
        
        self.is_fitted = True
        logger.info("Model training completed!")
        
        return self
    
    def predict(self, texts, image_paths):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        tfidf_features, sentiment_features = self.extract_text_features(texts, fit=False)
        image_features = self.extract_image_features(image_paths)
        
        # Convert and transform
        tfidf_dense = tfidf_features.toarray()
        
        if self.use_pca:
            tfidf_dense = self.pca_text.transform(tfidf_dense)
        
        tfidf_scaled = self.text_scaler.transform(tfidf_dense)
        image_scaled = self.image_scaler.transform(image_features)
        sentiment_scaled = self.sentiment_scaler.transform(sentiment_features)
        
        # Combine features with same OPTIMIZED weights as training
        combined_features = np.hstack([
            tfidf_scaled * 2.0,
            sentiment_scaled * 1.5,
            image_scaled * 1.2
        ])
        
        # Predict
        predictions = self.svm_classifier.predict(combined_features)
        probabilities = self.svm_classifier.predict_proba(combined_features)
        
        return predictions, probabilities
    
    def evaluate(self, texts, image_paths, labels):
        """Evaluate the model"""
        predictions, probabilities = self.predict(texts, image_paths)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        report = classification_report(labels, predictions, zero_division=0)
        
        return accuracy, report
    
    def save_model(self, filepath):
        """Save the model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'text_scaler': self.text_scaler,
            'image_scaler': self.image_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'pca_text': self.pca_text if self.use_pca else None,
            'svm_classifier': self.svm_classifier,
            'text_preprocessor': self.text_preprocessor,
            'image_extractor': self.image_extractor,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'random_state': self.random_state,
            'use_pca': self.use_pca
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model"""
        model_data = joblib.load(filepath)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.text_scaler = model_data['text_scaler']
        self.image_scaler = model_data['image_scaler']
        self.sentiment_scaler = model_data['sentiment_scaler']
        self.pca_text = model_data['pca_text']
        self.svm_classifier = model_data['svm_classifier']
        self.text_preprocessor = model_data['text_preprocessor']
        self.image_extractor = model_data['image_extractor']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.random_state = model_data['random_state']
        self.use_pca = model_data['use_pca']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")

def load_absa_data(data_dir):
    """Load data from TSV format"""
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
                        'label': int(parts[1]),
                        'tweet_id': parts[0],
                        'image_id': parts[2],
                        'text': parts[3],
                        'target': parts[4]
                    })
        return data
    
    train_data = load_tsv(train_file)
    dev_data = load_tsv(dev_file)
    test_data = load_tsv(test_file)
    
    return train_data, dev_data, test_data

def prepare_image_paths(data, image_base_path):
    """Prepare image paths"""
    image_paths = []
    for item in data:
        image_path = os.path.join(image_base_path, item['image_id'])
        image_paths.append(image_path)
    return image_paths

if __name__ == "__main__":
    # Configuration
    data_dir = "./absa_data/twitter2015"
    image_base_path = "./absa_data/twitter2015_images"
    
    # Load data
    logger.info("Loading data...")
    train_data, dev_data, test_data = load_absa_data(data_dir)
    
    # Prepare data
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    train_image_paths = prepare_image_paths(train_data, image_base_path)
    
    dev_texts = [item['text'] for item in dev_data]
    dev_labels = [item['label'] for item in dev_data]
    dev_image_paths = prepare_image_paths(dev_data, image_base_path)
    
    # Initialize classifier (set use_grid_search=True for parameter tuning)
    classifier = TFIDFSVMClassifier(
        max_features=30000,
        ngram_range=(1, 3),
        use_grid_search=False  # Set to True for best results (slower)
    )
    
    # Train model
    classifier.fit(train_texts, train_image_paths, train_labels)
    
    # Evaluate on dev set
    dev_accuracy, dev_report = classifier.evaluate(dev_texts, dev_image_paths, dev_labels)
    
    print("\n" + "="*60)
    print("ENHANCED TF-IDF + SVM RESULTS")
    print("="*60)
    print(f"Development Set Accuracy: {dev_accuracy:.4f}")
    print("\nClassification Report:")
    print(dev_report)
    print("="*60)
    
    # Save model
    classifier.save_model("enhanced_tfidf_svm_model.pkl")