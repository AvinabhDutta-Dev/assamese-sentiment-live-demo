import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_ID = "AvinabhDutta-Dev/assamese-sentiment-muril"

print("Loading model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
model.eval()

# Quantize to int8 for a much smaller memory footprint on CPU
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Model loaded successfully.")

LABELS = {0: "Negative", 1: "Positive"}  # confirm this matches your training label order

def predict_sentiment(text):
    if not text.strip():
        return {"Please enter some Assamese text.": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Assamese Review Text",
        placeholder="আপোনাৰ অসমীয়া পাঠ ইয়াত দিয়ক...",
        lines=3,
    ),
    outputs=gr.Label(num_top_classes=2, label="Sentiment Confidence"),
    title="Assamese Sentiment Analysis — MuRIL (96.33% Accuracy)",
    description=(
        "Fine-tuned MuRIL model trained on an original, dual-annotated 8,986-review "
        "Assamese movie review dataset. Part of a 5-model comparative study "
        "(SVM, CNN-BiLSTM, IndicBERT, XLM-RoBERTa, MuRIL)."
    ),
    examples=[
        ["এই নাটকখন সঁচাকৈয়ে ধুনীয়া আছিল, অভিনয়বোৰো বৰ ভাল লাগিল।"],
        ["সময় নষ্ট। কাহিনীটো একেবাৰে বিৰক্তিকৰ।"],
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
