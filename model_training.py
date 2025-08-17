import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# NLTK data (stopwords, punkt, wordnet) should be downloaded manually.

# 1. Read the dataset
print("Reading the dataset...")
df = pd.read_csv('cyberbullying_tweets(ML).csv')

# Rename columns for easier access
df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'label'}, inplace=True)
print("Dataset loaded successfully.")
print("Dataset Info:")
df.info()
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 2. Handle null values
print("\nChecking for null values...")
print(df.isnull().sum())
# No null values to handle based on initial exploration, but if there were:
# df.dropna(inplace=True)

# 3. Preprocess the text data
print("\nPreprocessing text data...")
lemmatizer = WordNetLemmatizer()
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
    # Remove stop words and apply lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete.")
print("\nFirst 5 rows with processed text:")
print(df[['text', 'processed_text']].head())

# 4. Transform words into vectors using TF-IDF Vectorizer
print("\nVectorizing text data using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']
print("Text vectorization complete.")

# 5. Select features (X) and target (y) - Done in the previous step

# 6. Split data into training and test data
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data splitting complete.")

# 7. Apply models and evaluate
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Linear SVM": LinearSVC()
}

best_model = None
best_accuracy = 0.0

for name, model in models.items():
    print(f"\n--- Training and evaluating {name} ---")
    # Train the model
    model.fit(X_train, y_train)
    
    # 8. Predict the cyberbullying type for test data
    y_pred = model.predict(X_test)
    
    # 9. Compute Confusion matrix and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 10. Report the model with the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest performing model is {best_model_name} with an accuracy of {best_accuracy:.4f}")

# Save the best model and the vectorizer
print("\nSaving the best model and the TF-IDF vectorizer...")
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully as 'best_model.pkl' and 'vectorizer.pkl'.")
