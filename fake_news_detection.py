import re
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the LAIR dataset
dataset = load_dataset("liar")

# Convert the Dataset object to a list of statements
train_statements = list(dataset['train']['statement'])
test_statements = list(dataset['test']['statement'])


# Preprocess the data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)  # Remove non-alphanumeric characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text


# Apply preprocessing to the lists of statements
train_statements_preprocessed = [preprocess_text(text) for text in train_statements]
test_statements_preprocessed = [preprocess_text(text) for text in test_statements]

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    train_statements_preprocessed, dataset['train']['label'], test_size=0.2, random_state=42
)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train the PassiveAggressiveClassifier model
model = PassiveAggressiveClassifier()
model.fit(X_train_tfidf, y_train)
accuracy = model.score(X_test_tfidf, y_test)

print("Model Accuracy:", accuracy)


pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
