# AI-Powered VoiceBot

This is a **speech-to-speech AI VoiceBot** for customer support. It can understand spoken user queries, detect the intent, generate a response, and speak back to the user.

# python version used
    python 3.11

## Features

- **Speech Recognition:** Converts user voice to text using **OpenAI Whisper**.
- **Intent Classification:** Detects user intent using a fine-tuned **BERT model**.
- **Response Generation:** Generates appropriate responses for customer queries.
- **Text-to-Speech:** Converts response text back to audio.
- **API:** FastAPI-based endpoint for handling audio requests.
# AI Voice Bot for Customer Support Automation

## Overview

This project is an AI-powered voice bot designed to automate customer support interactions.
The system accepts a user's voice input, converts it into text, identifies the user's intent, and generates an appropriate response which is then converted back into speech.

The goal of the project is to demonstrate how modern AI technologies can be combined to build a simple **end-to-end conversational voice system**.

---

## Features

* Speech-to-Text conversion
* Intent detection using a transformer-based NLP model
* Automated response generation
* Text-to-Speech output
* REST API built with FastAPI
* Simple modular architecture for easy extension

---

## System Architecture

The voice bot pipeline works as follows:

1. User uploads voice input
2. Speech is converted to text
3. Intent classification predicts the user's request
4. A response is generated based on the detected intent
5. Response text is converted into speech
6. Audio response is returned to the user


## Technologies Used

* Python
* FastAPI
* PyTorch
* Transformers
* Speech recognition models
* Text-to-Speech synthesis

The intent classification model uses **BERT** through the **Hugging Face Transformers** library.

Speech recognition is implemented using **OpenAI Whisper**.

---

## Project Structure

```
.
├── app.py                  # FastAPI main server
├── asr/
│   └── whisper_asr.py      # Speech recognition
├── data/
│   └── intents_dataset.csv # Intents Dataset
├── nlp/
│   └── intent_model.py     # BERT-based intent classifier
│   └──train_intent.py      # Train file
├── response/
│   └── response_mapper.py  # Maps intents to responses
├── tts/
│   └── tts_engine.py       # Text-to-speech engine
├── evaluation/
│   ├── intent_evaluation.py
│   └── wer_test.py
├── models/                 # Trained models & tokenizer
├── demo/                   # Demo Video
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/AI-voice-bot-for-customer-support-automation.git
```

Move into the project directory:

```
cd AI-voice-bot-for-customer-support-automation
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server:

```
uvicorn app:app --reload
```

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

From there you can upload an audio file and receive the generated response.

---

## Model Training

The intent classifier is trained using a transformer-based text classification model.

### Dataset

The dataset contains customer queries mapped to the following intents:

* order_status
* cancel_order
* refund_request
* payment_issue
* change_address
* delivery_delay
* complaint
* speak_to_agent

Each intent contains multiple example queries to help the model learn patterns in user requests.

### Training Process

The training pipeline performs the following steps:

1. Load the dataset
2. Convert intent labels to numeric values
3. Tokenize text using a pretrained tokenizer
4. Fine-tune a transformer model for classification
5. Save the trained model and tokenizer

Run training with:

```
python train_intent_model.py
```

After training, the model is saved in the `models/` directory.

---

## Example Usage

User input (voice):

```
"I want to cancel my order"
```

Predicted intent:

```
cancel_order
```

Bot response:

```
Your order cancellation request has been processed.
```

The response is returned as an audio file.

---

## Future Improvements

* Multi-language support
* More advanced dialogue management
* Integration with real customer service databases
* Deployment using Docker or cloud platforms
* Real-time microphone input

---

## Author

Rishabh Shandil

Machine Learning Enthusiast interested in building practical AI systems.
