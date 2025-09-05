from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class EcommerceRecommender:
    def __init__(self):
        self.vectorizer = None
        self.product_data = None
        self.tfidf_matrix = None
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.product_data = model_data['product_data']
            print("✅ Model loaded successfully!")
            print(f"📊 Total products: {len(self.product_data)}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def search_products(self, query, top_n=20):
        """Search for products based on query"""
        try:
            processed_query = self.preprocess_text(query)
            query_vec = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get all products with similarity > 0
            product_indices = np.where(similarities > 0.1)[0]
            
            # Sort by similarity score
            sorted_indices = product_indices[np.argsort(-similarities[product_indices])]
            
            results = []
            for idx in sorted_indices[:top_n]:
                product = self.product_data.iloc[idx].to_dict()
                product['similarity_score'] = float(similarities[idx])
                results.append(product)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

# Initialize Flask application
app = Flask(__name__)

# Initialize recommender
print("Initializing EcommerceRecommender...")
recommender = EcommerceRecommender()
model_loaded = recommender.load_model('ecommerce_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_products():
    try:
        query = request.args.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        if not model_loaded:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        print(f"🔍 Searching for: '{query}'")
        
        # Get search results
        results = recommender.search_products(query, top_n=20)
        
        print(f"📈 Found {len(results)} results")
        
        # Format results for frontend
        formatted_results = []
        for product in results:
            formatted_results.append({
                'product_id': product['product_id'],
                'name': product['product_name'],
                'price': f"₹{product['price']:,}",
                'category': product['category'],
                'sub_category': product['sub_category'],
                'brand': product['brand'],
                'score': product['similarity_score']
            })
        
        return jsonify({
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'total_products': len(recommender.product_data) if model_loaded else 0
    })

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check what's loaded"""
    if model_loaded:
        sample_products = recommender.product_data.head(3).to_dict('records')
        return jsonify({
            'model_loaded': True,
            'total_products': len(recommender.product_data),
            'sample_products': sample_products,
            'columns': list(recommender.product_data.columns)
        })
    else:
        return jsonify({'model_loaded': False})

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("🌐 Server will be available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)