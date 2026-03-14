import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:

    def __init__(self):

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

        # Load trained intent model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "models/intent_model"
        )

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Load label mapping from trained model
        self.id2label = self.model.config.id2label

        # Convert keys to integers (important)
        self.id2label = {int(k): v for k, v in self.id2label.items()}

    def predict(self, text):

        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Disable gradients for faster inference
        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)

            confidence, pred_id = torch.max(probs, dim=1)

            confidence = confidence.item()
            pred_id = pred_id.item()

        intent = self.id2label[pred_id]

        return intent, confidence