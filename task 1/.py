import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import ssl

# Fix SSL certificate issues on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

# Download required NLTK data
print("📚 Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("✅ NLTK data ready!")

class SpamClassifier:
    def __init__(self, use_stemming=True):
        """Initialize the Spam Classifier"""
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.models = {}
        
    def preprocess_text(self, text):
        """Preprocess the text: lowercase, remove punctuation, stop words, and apply stemming/lemmatization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and empty tokens
        tokens = [token for token in tokens if token not in self.stop_words and token.strip()]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare the SMS Spam Collection dataset"""
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Display dataset info
        print(f"\n📊 Dataset Information:")
        print(f"   Total messages: {len(df)}")
        print(f"   Spam messages: {df['label'].sum()}")
        print(f"   Ham messages: {len(df) - df['label'].sum()}")
        print(f"   Spam percentage: {(df['label'].sum() / len(df) * 100):.2f}%")
        
        # Preprocess messages
        print("\n🔄 Preprocessing text data...")
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        return df['processed_message'], df['label']
    
    def vectorize_text(self, X_train, X_test, method='tfidf', max_features=5000):
        """Vectorize text features using TF-IDF or Bag of Words"""
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                              ngram_range=(1, 2),
                                              min_df=2)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features,
                                             ngram_range=(1, 2),
                                             min_df=2)
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"   Vectorization method: {method.upper()}")
        print(f"   Number of features: {X_train_vec.shape[1]}")
        
        return X_train_vec, X_test_vec
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
            'Support Vector Machine': SVC(kernel='linear', C=1.0, probability=True)
        }
        
        print("\n🤖 Training models...")
        for name, model in models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        print("   ✅ All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            print(f"\n   {name}:")
            print(f"   ├─ Accuracy:  {accuracy:.4f}")
            print(f"   ├─ Precision: {precision:.4f}")
            print(f"   ├─ Recall:    {recall:.4f}")
            print(f"   └─ F1-Score:  {f1:.4f}")
            
        return results
    
    def plot_confusion_matrices(self, results, y_test):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(results.items()):
            conf_matrix = metrics['confusion_matrix']
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], 
                       yticklabels=['Ham', 'Spam'],
                       ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def predict_single_message(self, text, model_name='Logistic Regression'):
        """Predict if a single message is spam or not"""
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        model = self.models[model_name]
        prediction = model.predict(text_vec)[0]
        
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(text_vec)[0]
        
        return prediction, probability
    
    def create_prediction_function(self, model_name='Logistic Regression'):
        """Create a user-friendly prediction function"""
        def predict_spam(text):
            if not text or len(text.strip()) == 0:
                return "⚠️ Please provide a valid message"
            
            prediction, probability = self.predict_single_message(text, model_name)
            
            if prediction == 1:
                result = "🚨 SPAM"
                if probability is not None:
                    spam_prob = probability[1] * 100
                    result += f" (Confidence: {spam_prob:.2f}%)"
            else:
                result = "✅ NOT SPAM"
                if probability is not None:
                    ham_prob = probability[0] * 100
                    result += f" (Confidence: {ham_prob:.2f}%)"
            
            return result
        
        return predict_spam

def main():
    """Main function to run the spam classifier"""
    print("\n" + "="*60)
    print("📧 SMS SPAM CLASSIFIER")
    print("="*60)
    
    # Initialize the spam classifier
    classifier = SpamClassifier(use_stemming=True)
    
    # Load and prepare data
    try:
        X, y = classifier.load_and_prepare_data('spam.csv')
    except FileNotFoundError:
        print("\n❌ Error: spam.csv not found!")
        print("Please run: python download_data.py")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(X_train)} messages")
    print(f"   Testing set: {len(X_test)} messages")
    
    # Vectorize text features
    print("\n🔤 Vectorizing text features...")
    X_train_vec, X_test_vec = classifier.vectorize_text(X_train, X_test, method='tfidf')
    
    # Train models
    classifier.train_models(X_train_vec, y_train)
    
    # Evaluate models
    results = classifier.evaluate_models(X_test_vec, y_test)
    
    # Plot confusion matrices
    print("\n📊 Generating confusion matrices...")
    classifier.plot_confusion_matrices(results, y_test)
    
    # Create prediction function
    predict_spam = classifier.create_prediction_function(model_name='Logistic Regression')
    
    # Test with custom messages
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTION FUNCTION")
    print("="*60)
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
        "Hey, are we still meeting for lunch tomorrow at 12?",
        "URGENT: Your account has been compromised. Call this number immediately!",
        "Don't forget to pick up milk on your way home.",
        "You have been selected for an exclusive offer! Reply NOW to claim $1000!",
        "The meeting has been rescheduled to 3 PM."
    ]
    
    for i, msg in enumerate(test_messages, 1):
        result = predict_spam(msg)
        print(f"\n{i}. Message: {msg}")
        print(f"   Prediction: {result}")
    
    print("\n" + "="*60)
    print("✅ Classification complete!")
    print("="*60)

if __name__ == "__main__":
    main()