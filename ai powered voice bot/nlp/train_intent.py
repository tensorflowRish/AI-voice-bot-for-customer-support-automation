import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/intents_dataset.csv")

label_dict = {label: i for i, label in enumerate(df.intent.unique())}
df["label"] = df.intent.map(label_dict)

train_df, test_df = train_test_split(df, test_size=0.2)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

labels = [
    "cancel_order",
    "change_address",
    "complaint",
    "delivery_delay",
    "order_status",
    "payment_issue",
    "refund_request",
    "speak_to_agent"
]

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model("./models/intent_model")
tokenizer.save_pretrained("./models/tokenizer")