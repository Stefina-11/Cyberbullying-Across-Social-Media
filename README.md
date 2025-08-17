# Cyberbullying-Across-Social-Media

This project is a web application that uses a machine learning model to detect cyberbullying in text. The backend is a Flask server that provides an API for the model, and the frontend is a React application that allows users to enter text and see the model's prediction.

## Model

The project uses a TF-IDF vectorizer to process the text data. Several machine learning models are trained and evaluated, including:
- Multinomial Naive Bayes
- Logistic Regression
- Decision Tree
- Linear SVM

The best-performing model is saved and used for predictions.

## Technologies Used

### Backend
- Python
- Flask
- scikit-learn
- pandas
- numpy
- nltk
- joblib

### Frontend
- React
- JavaScript
- HTML
- CSS

## Running the Project

### Prerequisites

- Python 3.x
- Node.js and npm

### Backend

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Stefina-11/Cyberbullying-Across-Social-Media.git
   cd Cyberbullying-Across-Social-Media
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```

### Frontend

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the React app:**
   ```bash
   npm start
   ```
