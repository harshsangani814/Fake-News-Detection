from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


# Preprocess the text (same as in the backend)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)  # Remove non-alphanumeric characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news = preprocess_text(news)  # Apply the preprocessing
    news_tfidf = vectorizer.transform([news])
    prediction = model.predict(news_tfidf)

    # Mapping for prediction labels
    label_map = {
        0: 'false',
        1: 'half-true',
        2: 'mostly-true',
        3: 'true',
        4: 'barely-true',
        5: 'pants-fire'
    }

    # Get the textual label using the mapping
    textual_label = label_map[prediction[0]]

    return render_template('result.html', prediction=textual_label)


if __name__ == '__main__':
    app.run(debug=True)
