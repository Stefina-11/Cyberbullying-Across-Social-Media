from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize words
    tokens = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            text = data['text']
            
            if not text:
                return jsonify({'error': 'Text input is empty'}), 400
            
            # Preprocess the input text
            processed_text = preprocess_text(text)
            
            # Vectorize the processed text
            vectorized_text = vectorizer.transform([processed_text])
            
            # Make a prediction
            prediction = model.predict(vectorized_text)
            
            # Return the prediction as JSON
            return jsonify({'prediction': prediction[0]})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
