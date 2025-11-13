# coding=utf-8
"""
Implementasi metode klasik TF-IDF + SVM untuk analisis sentimen multimodal (ULTRA OPTIMIZED)
Target: 90%+ accuracy, training selama 3+ hari
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import cv2
from PIL import Image
import pickle
import joblib
from tqdm import tqdm
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from textblob import TextBlob
import nltk
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraTextPreprocessor:
    """Ultra advanced text preprocessing for maximum feature extraction"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = SnowballStemmer('english')
        except:
            self.lemmatizer = None
            self.stemmer = None
        
        # Extended sentiment lexicons
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                                   'love', 'best', 'perfect', 'beautiful', 'awesome', 'brilliant',
                                   'outstanding', 'superb', 'magnificent', 'marvelous', 'terrific',
                                   'fabulous', 'incredible', 'spectacular', 'phenomenal', 'exceptional',
                                   'remarkable', 'impressive', 'delightful', 'charming', 'lovely'])
        
        self.negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                                   'poor', 'disappointing', 'useless', 'waste', 'disgusting',
                                   'pathetic', 'dreadful', 'appalling', 'abysmal', 'atrocious',
                                   'deplorable', 'detestable', 'revolting', 'repulsive', 'vile',
                                   'loathsome', 'despicable', 'contemptible', 'offensive', 'annoying'])
        
        # Intensifiers and negations
        self.intensifiers = set(['very', 'extremely', 'incredibly', 'absolutely', 'totally',
                                'completely', 'utterly', 'highly', 'quite', 'really', 'truly'])
        self.negations = set(['not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody',
                             'none', 'cannot', 'cant', 'wont', 'wouldnt', 'shouldnt', 'dont'])
    
    def preprocess(self, text):
        """Ultra advanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle contractions
        contractions = {
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', ' ', text)
        
        # Handle repeated characters (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_advanced_sentiment_features(self, text):
        """Extract comprehensive sentiment-based features"""
        words = text.lower().split()
        
        # Basic sentiment counts
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Intensifier and negation analysis
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        negation_count = sum(1 for word in words if word in self.negations)
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        comma_count = text.count(',')
        period_count = text.count('.')
        
        # Text statistics
        text_length = len(text)
        word_count = len(words)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Capital letters analysis
        capital_count = sum(1 for c in text if c.isupper())
        capital_ratio = capital_count / len(text) if len(text) > 0 else 0
        
        # Unique word ratio
        unique_words = len(set(words))
        unique_ratio = unique_words / word_count if word_count > 0 else 0
        
        # TextBlob sentiment (if available)
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except:
            polarity = 0
            subjectivity = 0
        
        # Combine all features
        features = np.array([
            pos_count, neg_count, intensifier_count, negation_count,
            exclamation_count, question_count, comma_count, period_count,
            text_length, word_count, sentence_count, avg_word_length, avg_sentence_length,
            capital_count, capital_ratio, unique_words, unique_ratio,
            polarity, subjectivity,
            # Additional ratios
            pos_count / word_count if word_count > 0 else 0,
            neg_count / word_count if word_count > 0 else 0,
            (pos_count - neg_count) / word_count if word_count > 0 else 0
        ])
        
        return features

class UltraImageFeatureExtractor:
    """Ultra enhanced image feature extraction"""
    
    def __init__(self):
        self.target_size = (256, 256)  # Increased for better features
    
    def extract_features(self, img_path):
        """Extract ultra comprehensive image features"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return np.zeros(500)  # Increased feature size
                
            img = cv2.resize(img, self.target_size)
                
            # Multiple feature types
            color_features = self._extract_ultra_color_features(img)
            texture_features = self._extract_ultra_texture_features(img)
            edge_features = self._extract_ultra_edge_features(img)
            statistical_features = self._extract_ultra_statistical_features(img)
            shape_features = self._extract_shape_features(img)
            
            # Combine all features
            all_features = np.concatenate([
                color_features,
                texture_features, 
                edge_features,
                statistical_features,
                shape_features
            ])
            
            # Ensure fixed size
            if len(all_features) < 500:
                all_features = np.pad(all_features, (0, 500 - len(all_features)), 'constant')
            else:
                all_features = all_features[:500]
            
            return all_features
                
        except Exception as e:
            logger.warning(f"Error extracting features from {img_path}: {e}")
            return np.zeros(500)
    
    def _extract_ultra_color_features(self, img):
        """Ultra enhanced color feature extraction"""
        # Multiple color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        features = []
        
        # HSV histograms with more bins
        # H channel: [0, 180], S and V channels: [0, 256]
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        for hist in [hist_h, hist_s, hist_v]:
            hist = hist.flatten() / (hist.sum() + 1e-7)
            features.extend(hist)
        
        # Color moments for multiple color spaces
        for color_space in [img, hsv, lab, yuv]:
            for channel in cv2.split(color_space):
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.var(channel),
                    np.median(channel)
                ])
        
        # Dominant colors analysis
        pixels = img.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        color_variance = np.var(pixels, axis=0)
        features.extend(dominant_color)
        features.extend(color_variance)
        
        return np.array(features)
    
    def _extract_ultra_texture_features(self, img):
        """Ultra enhanced texture features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # LBP with multiple radii
        for radius in [1, 2, 3]:
            lbp = self._compute_lbp_radius(gray, radius)
            hist_lbp, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
            hist_lbp = hist_lbp / (hist_lbp.sum() + 1e-7)
            features.extend(hist_lbp)
        
        # Gabor filters with multiple orientations and frequencies
        for theta in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
            for freq in [0.1, 0.3, 0.5]:
                kernel = cv2.getGaborKernel((7, 7), 2.0, theta, 10.0, freq, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                features.extend([np.mean(filtered), np.std(filtered), np.var(filtered)])
        
        return np.array(features)
    
    def _compute_lbp_radius(self, gray, radius):
        """Compute LBP with variable radius"""
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0]-radius):
            for j in range(radius, gray.shape[1]-radius):
                center = gray[i, j]
                code = 0
                # 8 neighbors
                neighbors = [
                    gray[i-radius, j-radius], gray[i-radius, j], gray[i-radius, j+radius],
                    gray[i, j+radius], gray[i+radius, j+radius], gray[i+radius, j],
                    gray[i+radius, j-radius], gray[i, j-radius]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        return lbp
    
    def _extract_ultra_edge_features(self, img):
        """Ultra enhanced edge features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Multiple edge detection methods
        # Canny with different thresholds
        for low, high in [(50, 150), (100, 200), (30, 100)]:
            edges = cv2.Canny(gray, low, high)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
        
        # Sobel in different directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        features.extend([
            np.mean(gradient_magnitude), np.std(gradient_magnitude), np.var(gradient_magnitude),
            np.mean(gradient_direction), np.std(gradient_direction)
        ])
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.mean(laplacian), np.std(laplacian), np.var(laplacian)])
        
        return np.array(features)
    
    def _extract_ultra_statistical_features(self, img):
        """Ultra enhanced statistical features"""
        features = []
        
        # Per channel statistics
        for channel in cv2.split(img):
            features.extend([
                np.mean(channel), np.std(channel), np.var(channel),
                np.median(channel), np.min(channel), np.max(channel),
                np.percentile(channel, 10), np.percentile(channel, 25),
                np.percentile(channel, 75), np.percentile(channel, 90),
                np.ptp(channel)  # peak-to-peak
            ])
        
        # Overall image statistics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.median(gray), np.min(gray), np.max(gray)
        ])
        
        return np.array(features)
    
    def _extract_shape_features(self, img):
        """Extract shape-based features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(20)
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        features = []
        
        # Basic shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
        else:
            compactness = 0
        
        features.extend([area, perimeter, compactness])
        
        # Hu moments
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform to handle large values
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments)
        else:
            features.extend([0] * 7)
        
        # Bounding rectangle features
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        
        features.extend([aspect_ratio, extent])
        
        # Ensure fixed length
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])

class UltraTFIDFSVMClassifier:
    """
    Ultra enhanced TF-IDF + SVM classifier for maximum accuracy
    """
    
    def __init__(self, max_features=200000, ngram_range=(1, 4), random_state=42, 
                 use_extensive_search=True, use_images=True, use_pca=True, 
                 use_feature_selection=True, use_ensemble=True, progress_callback=None):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.use_extensive_search = use_extensive_search
        self.use_images = use_images
        self.use_pca = use_pca
        self.use_feature_selection = use_feature_selection
        self.use_ensemble = use_ensemble
        self.progress_callback = progress_callback
        
        # Ultra text preprocessor
        self.text_preprocessor = UltraTextPreprocessor()
        
        # Multiple TF-IDF vectorizers for ensemble
        self.tfidf_vectorizers = [
            TfidfVectorizer(
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
                token_pattern=r'\b\w+\b',
                analyzer='word'
            ),
            TfidfVectorizer(
                max_features=max_features//2,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                min_df=1,
                max_df=0.9,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm='l1',
                token_pattern=r'\b\w+\b',
                analyzer='char_wb'
            )
        ]
        
        # Image feature extractor
        self.image_extractor = UltraImageFeatureExtractor()
        
        # Multiple scalers
        self.text_scalers = [StandardScaler(), RobustScaler(), MinMaxScaler()]
        self.image_scaler = RobustScaler()
        self.sentiment_scaler = StandardScaler()
        
        # Dimensionality reduction
        self.pca_text = PCA(n_components=0.98, random_state=random_state)
        self.svd_text = TruncatedSVD(n_components=1000, random_state=random_state)
        
        # Feature selection (will be set dynamically based on feature size)
        self.feature_selector = None
        self.feature_selection_k = 5000
        
        # Ensemble classifier
        if use_ensemble:
            self.classifiers = None  # Will be set after training
        else:
            self.svm_classifier = SVC(
                kernel='rbf',
                C=1000.0,
                gamma='scale',
                random_state=random_state,
                probability=True,
                class_weight='balanced',
                cache_size=5000,
                decision_function_shape='ovo',
                shrinking=True,
                tol=1e-4
            )
        
        self.is_fitted = False

    def _update_progress(self, phase, current, total, message=None):
        """Send progress updates if callback is provided"""
        if self.progress_callback is None:
            return

        try:
            if total <= 0:
                percent = 0.0
            else:
                percent = max(0.0, min(100.0, (current / total) * 100.0))

            self.progress_callback(
                phase=phase,
                percent=percent,
                current=current,
                total=total,
                message=message or ""
            )
        except Exception as callback_error:
            logger.debug(f"Progress callback error ignored: {callback_error}")
    
    def _preprocess_texts(self, texts):
        """Preprocess all texts"""
        return [self.text_preprocessor.preprocess(text) for text in texts]
    
    def _extract_sentiment_features(self, texts):
        """Extract sentiment features from texts"""
        features = []
        for text in texts:
            features.append(self.text_preprocessor.extract_advanced_sentiment_features(text))
        return np.array(features)
    
    def extract_text_features(self, texts, fit=True):
        """Extract ultra enhanced text features"""
        logger.info("Extracting ultra enhanced text features...")
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Multiple TF-IDF features
        tfidf_features_list = []
        for i, vectorizer in enumerate(self.tfidf_vectorizers):
            if fit:
                features = vectorizer.fit_transform(processed_texts)
            else:
                features = vectorizer.transform(processed_texts)
            tfidf_features_list.append(features)
        
        # Combine TF-IDF features
        combined_tfidf = hstack(tfidf_features_list)
        
        # Sentiment features
        sentiment_features = self._extract_sentiment_features(processed_texts)
        
        return combined_tfidf, sentiment_features
    
    def extract_image_features(self, image_paths, phase="train_image_extraction"):
        """Extract ultra enhanced image features"""
        if not self.use_images:
            return np.zeros((len(image_paths), 500))
        
        logger.info(f"Extracting ultra enhanced features from {len(image_paths)} images...")
        features = []
        total_images = len(image_paths)
        self._update_progress(phase, 0, total_images, "Starting image feature extraction")
        
        for idx, img_path in enumerate(image_paths, start=1):
            features.append(self.image_extractor.extract_features(img_path))
            self._update_progress(
                phase,
                idx,
                total_images,
                f"Processing images ({idx}/{total_images})"
            )

        self._update_progress(phase, total_images, total_images, "Image feature extraction completed")
        
        return np.array(features)
    
    def _perform_extensive_search(self, X_train, y_train):
        """Perform extensive hyperparameter search"""
        logger.info("Performing extensive hyperparameter search (this will take a long time)...")
        self._update_progress("hyperparameter_search", 0, 100, "Starting extensive search")
        
        # Extended parameter grid for maximum accuracy
        param_distributions = {
            'C': [0.1, 1, 10, 100, 1000, 5000, 10000],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4, 5],  # for poly kernel
            'class_weight': ['balanced', None]
        }
        
        svm = SVC(random_state=self.random_state, probability=True, cache_size=5000)
        
        # Use RandomizedSearchCV for efficiency with extensive search
        search = RandomizedSearchCV(
            svm, param_distributions, 
            n_iter=200,  # More iterations for better results
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state),
            scoring='f1_macro', 
            n_jobs=-1, 
            verbose=0,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Progress tracking for joblib parallel execution (per-fit)
        try:
            import joblib
            from joblib import parallel as joblib_parallel
            original_callback = joblib_parallel.BatchCompletionCallBack
            
            # Total fits = n_iter * n_splits
            if hasattr(search, 'cv') and hasattr(search.cv, 'get_n_splits'):
                n_splits = search.cv.get_n_splits(X_train, y_train)
            else:
                # Fallback split count
                n_splits = 10
            total_fits = int(search.n_iter * n_splits)
            completed_fits = {'count': 0}
            
            outer_update = self._update_progress
            
            class ProgressBatchCallback(original_callback):
                def __call__(self, *args, **kwargs):
                    result = super().__call__(*args, **kwargs)
                    try:
                        # batch_size attribute indicates how many tasks just completed
                        batch = getattr(self, 'batch_size', 1)
                        completed_fits['count'] += int(batch)
                        # Clamp to total
                        current = min(completed_fits['count'], total_fits)
                        message = f"Hyperparameter search progress ({current}/{total_fits} fits)"
                        outer_update("hyperparameter_search", current, total_fits, message)
                    except Exception:
                        pass
                    return result
            
            # Monkey patch joblib callback
            joblib_parallel.BatchCompletionCallBack = ProgressBatchCallback
            try:
                search.fit(X_train, y_train)
            finally:
                # Restore original callback
                joblib_parallel.BatchCompletionCallBack = original_callback
        except Exception:
            # If progress hook fails, fall back to normal fit
            search.fit(X_train, y_train)
        
        self._update_progress("hyperparameter_search", 100, 100, "Hyperparameter search completed")
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def _create_ensemble(self, X_train, y_train):
        """Create ensemble of classifiers"""
        logger.info("Creating ensemble of classifiers...")
        self._update_progress("ensemble_training", 0, 100, "Initializing ensemble training")
        
        # If extensive search is enabled, use it for the main SVM
        if self.use_extensive_search:
            logger.info("Performing extensive search for ensemble SVM...")
            base_svm = self._perform_extensive_search(X_train, y_train)
        else:
            base_svm = SVC(kernel='rbf', C=1000, gamma='scale', probability=True, 
                          random_state=self.random_state, class_weight='balanced')
        
        # Base classifiers with different strengths
        classifiers = [
            ('svm_rbf', base_svm),
            ('svm_poly', SVC(kernel='poly', degree=3, C=100, probability=True,
                            random_state=self.random_state, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=20, 
                                        random_state=self.random_state, 
                                        class_weight='balanced', n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                            random_state=self.random_state)),
            ('lr', LogisticRegression(C=100, random_state=self.random_state,
                                    class_weight='balanced', max_iter=2000))
        ]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=classifiers,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        logger.info("Training ensemble classifiers (this may take a long time)...")
        ensemble.fit(X_train, y_train)
        self._update_progress("ensemble_training", 100, 100, "Ensemble training completed")
        
        return ensemble
    
    def fit(self, texts, image_paths, labels):
        """Train the ultra model"""
        logger.info("Training ultra enhanced TF-IDF + SVM model...")
        pipeline_steps = 6  # text, images, reduction, scaling, fusion, classifier
        if self.use_feature_selection:
            pipeline_steps += 1
        step = 0
        self._update_progress("fit_pipeline", step, pipeline_steps, "Starting training pipeline")
        
        # Extract text features
        tfidf_features, sentiment_features = self.extract_text_features(texts, fit=True)
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Text features extracted")
        
        # Extract image features
        image_features = self.extract_image_features(image_paths, phase="train_image_extraction")
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Image features extracted")
        
        # Convert sparse to dense
        tfidf_dense = tfidf_features.toarray()
        
        # Apply dimensionality reduction to text features
        if self.use_pca:
            logger.info("Applying PCA to text features...")
            tfidf_pca = self.pca_text.fit_transform(tfidf_dense)
            tfidf_svd = self.svd_text.fit_transform(tfidf_dense)
            # Combine PCA and SVD features
            tfidf_reduced = np.hstack([tfidf_pca, tfidf_svd])
            logger.info(f"Text features reduced to {tfidf_reduced.shape[1]} dimensions")
        else:
            tfidf_reduced = tfidf_dense
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Dimensionality reduction applied")
        
        # Scale features with multiple scalers and combine
        text_scaled_list = []
        for scaler in self.text_scalers:
            text_scaled_list.append(scaler.fit_transform(tfidf_reduced))
        text_scaled = np.hstack(text_scaled_list)
        
        image_scaled = self.image_scaler.fit_transform(image_features) if self.use_images else image_features
        sentiment_scaled = self.sentiment_scaler.fit_transform(sentiment_features)
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Features scaled")
        
        # Ultra optimized weighted fusion for maximum accuracy
        if self.use_images:
            combined_features = np.hstack([
                text_scaled * 3.0,      # Increased text weight
                sentiment_scaled * 2.0,  # Increased sentiment weight
                image_scaled * 1.5       # Increased image weight
            ])
        else:
            combined_features = np.hstack([
                text_scaled * 3.0,
                sentiment_scaled * 2.0
            ])
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Feature fusion completed")
        
        # Feature selection for best features
        if self.use_feature_selection:
            logger.info("Performing feature selection...")
            # Dynamically set k based on feature size
            n_features = combined_features.shape[1]
            k = min(self.feature_selection_k, n_features)
            if k < n_features:
                self.feature_selector = SelectKBest(f_classif, k=k)
                combined_features = self.feature_selector.fit_transform(combined_features, labels)
                logger.info(f"Selected {k} best features from {n_features} total features")
            else:
                logger.info(f"Feature selection skipped: {n_features} features <= {k} (selection threshold)")
                self.feature_selector = None
            step += 1
            self._update_progress("fit_pipeline", step, pipeline_steps, "Feature selection completed")
        
        # Train classifier(s)
        if self.use_ensemble:
            self.classifiers = self._create_ensemble(combined_features, labels)
        else:
            if self.use_extensive_search:
                self.svm_classifier = self._perform_extensive_search(combined_features, labels)
            else:
                self.svm_classifier.fit(combined_features, labels)
        step += 1
        self._update_progress("fit_pipeline", step, pipeline_steps, "Classifier training completed")
        
        self.is_fitted = True
        logger.info("Ultra model training completed!")
        
        return self
    
    def predict(self, texts, image_paths, phase="predict", batch_size=256):
        """Make predictions with batched progress updates"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        n = len(texts)
        preds_all = []
        probs_all = []

        total_batches = max(1, int(np.ceil(n / float(batch_size))))
        self._update_progress(f"{phase}_progress", 0, total_batches, "Starting batched prediction")

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(n, (batch_idx + 1) * batch_size)

            texts_batch = texts[start:end]
            images_batch = image_paths[start:end]

            # Extract features for batch
            tfidf_features, sentiment_features = self.extract_text_features(texts_batch, fit=False)
            image_features = self.extract_image_features(images_batch, phase=f"{phase}_image_extraction")

            # Convert and transform
            tfidf_dense = tfidf_features.toarray()

            if self.use_pca:
                tfidf_pca = self.pca_text.transform(tfidf_dense)
                tfidf_svd = self.svd_text.transform(tfidf_dense)
                tfidf_reduced = np.hstack([tfidf_pca, tfidf_svd])
            else:
                tfidf_reduced = tfidf_dense

            # Scale with multiple scalers
            text_scaled_list = []
            for scaler in self.text_scalers:
                text_scaled_list.append(scaler.transform(tfidf_reduced))
            text_scaled = np.hstack(text_scaled_list)

            image_scaled = self.image_scaler.transform(image_features) if self.use_images else image_features
            sentiment_scaled = self.sentiment_scaler.transform(sentiment_features)

            # Combine features with same weights as training
            if self.use_images:
                combined_features = np.hstack([
                    text_scaled * 3.0,
                    sentiment_scaled * 2.0,
                    image_scaled * 1.5
                ])
            else:
                combined_features = np.hstack([
                    text_scaled * 3.0,
                    sentiment_scaled * 2.0
                ])

            # Apply feature selection
            if self.use_feature_selection and self.feature_selector is not None:
                combined_features = self.feature_selector.transform(combined_features)

            # Predict batch
            if self.use_ensemble:
                batch_preds = self.classifiers.predict(combined_features)
                batch_probs = self.classifiers.predict_proba(combined_features)
            else:
                batch_preds = self.svm_classifier.predict(combined_features)
                batch_probs = self.svm_classifier.predict_proba(combined_features)

            preds_all.append(batch_preds)
            probs_all.append(batch_probs)

            self._update_progress(f"{phase}_progress", batch_idx + 1, total_batches, f"Batched prediction {batch_idx + 1}/{total_batches}")

        predictions = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.array([])
        probabilities = np.vstack(probs_all) if len(probs_all) > 0 else np.array([])

        return predictions, probabilities
    
    def evaluate(self, texts, image_paths, labels, phase="evaluation", batch_size=256):
        """Evaluate the model with progress updates"""
        self._update_progress(phase, 0, 100, "Starting evaluation")
        predictions, probabilities = self.predict(texts, image_paths, phase=phase, batch_size=batch_size)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        report = classification_report(labels, predictions, zero_division=0)
        self._update_progress(phase, 100, 100, "Evaluation completed")
        return accuracy, report
    
    def save_model(self, filepath):
        """Save the ultra model"""
        model_data = {
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'text_scalers': self.text_scalers,
            'image_scaler': self.image_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'pca_text': self.pca_text if self.use_pca else None,
            'svd_text': self.svd_text if self.use_pca else None,
            'feature_selector': self.feature_selector if (self.use_feature_selection and self.feature_selector is not None) else None,
            'classifiers': self.classifiers if self.use_ensemble else None,
            'svm_classifier': self.svm_classifier if not self.use_ensemble else None,
            'text_preprocessor': self.text_preprocessor,
            'image_extractor': self.image_extractor,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'random_state': self.random_state,
            'use_pca': self.use_pca,
            'use_feature_selection': self.use_feature_selection,
            'use_ensemble': self.use_ensemble,
            'feature_selection_k': self.feature_selection_k
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ultra model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the ultra model"""
        model_data = joblib.load(filepath)
        
        self.tfidf_vectorizers = model_data['tfidf_vectorizers']
        self.text_scalers = model_data['text_scalers']
        self.image_scaler = model_data['image_scaler']
        self.sentiment_scaler = model_data['sentiment_scaler']
        self.pca_text = model_data['pca_text']
        self.svd_text = model_data['svd_text']
        self.feature_selector = model_data.get('feature_selector', None)
        self.feature_selection_k = model_data.get('feature_selection_k', 5000)
        self.classifiers = model_data['classifiers']
        self.svm_classifier = model_data['svm_classifier']
        self.text_preprocessor = model_data['text_preprocessor']
        self.image_extractor = model_data['image_extractor']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.random_state = model_data['random_state']
        self.use_pca = model_data['use_pca']
        self.use_feature_selection = model_data['use_feature_selection']
        self.use_ensemble = model_data['use_ensemble']
        self.is_fitted = True
        
        logger.info(f"Ultra model loaded from {filepath}")

# Keep the same data loading functions
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
    # Ultra configuration for maximum accuracy
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
    
    # Initialize ultra classifier for maximum accuracy
    classifier = UltraTFIDFSVMClassifier(
        max_features=200000,      # Increased for better coverage
        ngram_range=(1, 4),       # Extended n-grams
        use_extensive_search=True, # Enable extensive hyperparameter search
        use_images=True,          # Use image features
        use_pca=True,             # Use dimensionality reduction
        use_feature_selection=True, # Use feature selection
        use_ensemble=True         # Use ensemble methods
    )
    
    # Train ultra model (will take a very long time)
    logger.info("Starting ultra training (this will take 3+ days)...")
    classifier.fit(train_texts, train_image_paths, train_labels)
    
    # Evaluate on dev set
    dev_accuracy, dev_report = classifier.evaluate(dev_texts, dev_image_paths, dev_labels)
    
    print("\n" + "="*60)
    print("ULTRA ENHANCED TF-IDF + SVM RESULTS")
    print("="*60)
    print(f"Development Set Accuracy: {dev_accuracy:.4f}")
    print("\nClassification Report:")
    print(dev_report)
    print("="*60)
    
    # Save ultra model
    classifier.save_model("ultra_tfidf_svm_model.pkl")