import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re

class EcommerceRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            min_df=1,
            max_df=0.8
        )
        self.product_data = None
        self.tfidf_matrix = None
        
    def load_data(self, filepath):
        """Load product data from CSV"""
        try:
            self.product_data = pd.read_csv(filepath)
            print(f"✅ Loaded {len(self.product_data)} products from {filepath}")
            
            # Convert related_accessories to string and handle NaN
            self.product_data['related_accessories'] = self.product_data['related_accessories'].fillna('').astype(str)
            
            # Create combined text for better search
            print("Creating search text for products...")
            self.product_data['search_text'] = self.product_data.apply(
                lambda row: f"{row['product_name']} {row['category']} {row['sub_category']} {row['brand']}", 
                axis=1
            )
            
            return self.product_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def train(self):
        """Train the recommendation model"""
        if self.product_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Preprocessing product data for training...")
        
        # Preprocess all text fields for better search
        product_texts = []
        for _, row in self.product_data.iterrows():
            text = self.preprocess_text(row['search_text'])
            product_texts.append(text)
        
        print("Training TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(product_texts)
        print(f"✅ Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
    
    def search_products(self, query, top_n=10):
        """Search for products based on query"""
        try:
            processed_query = self.preprocess_text(query)
            query_vec = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get all products with similarity > 0
            product_indices = np.where(similarities > 0.1)[0]
            
            # Sort by similarity score (descending)
            if len(product_indices) > 0:
                sorted_indices = product_indices[np.argsort(-similarities[product_indices])]
            else:
                return []
            
            results = []
            for idx in sorted_indices[:top_n]:
                product = self.product_data.iloc[idx].to_dict()
                product['similarity_score'] = float(similarities[idx])
                results.append(product)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'product_data': self.product_data
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        print(f"📊 Total products: {len(self.product_data)}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.product_data = model_data['product_data']
        return model_data

# Train and save the model
if __name__ == "__main__":
    print("🚀 Starting E-commerce Recommender Model Training")
    print("=" * 50)
    
    try:
        # Initialize and train the model
        recommender = EcommerceRecommender()
        
        print("📊 Loading data...")
        data = recommender.load_data('data/sample_products.csv')
        
        print("🤖 Training model...")
        recommender.train()
        
        print("💾 Saving model...")
        recommender.save_model('ecommerce_model.pkl')
        
        # Test the model
        print("\n🧪 Testing search functionality:")
        print("-" * 30)
        
        test_queries = ["samsung", "moto", "iphone", "watch", "charger"]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = recommender.search_products(query, top_n=3)
            
            if results:
                for product in results:
                    print(f"  ✅ {product['product_name']} (Score: {product['similarity_score']:.3f})")
            else:
                print("  ❌ No results found")
        
        print("\n🎉 Model training and testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()