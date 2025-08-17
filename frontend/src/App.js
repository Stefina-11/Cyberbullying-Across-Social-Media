import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setPrediction('');

    if (!text.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      if (data.prediction) {
        setPrediction(`Predicted Cyberbullying Type: ${data.prediction}`);
      } else if (data.error) {
        setError(`Error: ${data.error}`);
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Cyberbullying Prediction</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            rows="10"
            cols="50"
          />
          <br />
          <button type="submit">Predict</button>
        </form>
        {prediction && <p className="prediction">{prediction}</p>}
        {error && <p className="error">{error}</p>}
      </header>
    </div>
  );
}

export default App;
