import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from nlp.intent_model import IntentClassifier

# load model
model = IntentClassifier()

# load dataset
data = pd.read_csv("data/intents_dataset.csv")

y_true = data["intent"]
y_pred = []

# predict intents
for text in data["text"]:
    intent, confidence = model.predict(text)
    y_pred.append(intent)

# show metrics
print("Classification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))