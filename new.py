#Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from skl earn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 2: Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Step 3: Add labels
fake['label'] = 0  # Fake = 0
true['label'] = 1  # Real = 1

# Step 4: Combine and shuffle
data = pd.concat([fake[['text', 'label']], true[['text', 'label']]], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# Step 5: Preprocess text
stop_words = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Step 7: TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 8: Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Step 10: Predict your own news
def predict_article(text):
    cleaned = " ".join([word for word in text.split() if word.lower() not in stop_words])
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Try it:
your_article = "Breaking: Government introduces new education policy that will reduce exam stress."
print("Prediction:", predict_article(your_article))
