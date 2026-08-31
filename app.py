import os
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import torch
import time
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_ID = "AvinabhDutta-Dev/assamese-sentiment-muril-onnx"

print("Loading ONNX model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID)
print("Model loaded successfully.")

LABELS = {0: "Negative", 1: "Positive"}

# Static benchmark results from the full 5-model comparative study
MODEL_STATS = """
### 📊 Model Performance (from full evaluation)
| Metric | Score |
|---|---|
| Accuracy | 96.33% |
| Macro F1-score | 0.9605 |
| Macro Precision | 0.9608 |
| Macro Recall | 0.9602 |
| ROC-AUC | 0.9882 |

**Training data:** 8,986-review original Assamese dataset (dual-annotated, 97.33% inter-annotator agreement, Cohen's Kappa 0.942)

**Rank:** Best of 5 models tested (SVM, CNN-BiLSTM, IndicBERT, XLM-RoBERTa, MuRIL)
"""

def predict_sentiment(text):
    if not text.strip():
        return {}, "Please enter some Assamese text.", ""

    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    elapsed = (time.time() - start) * 1000  # ms

    result = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    token_count = len(inputs["input_ids"][0])
    details = (
        f"**Inference time:** {elapsed:.0f} ms  \n"
        f"**Tokens processed:** {token_count}  \n"
        f"**Model:** MuRIL (fine-tuned, ONNX runtime)"
    )

    return result, details, ""

with gr.Blocks(title="Assamese Sentiment Analysis") as demo:
    gr.Markdown("# Assamese Sentiment Analysis — MuRIL")
    gr.Markdown("Fine-tuned MuRIL model on an original, dual-annotated 8,986-review Assamese movie review dataset.")

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Assamese Review Text",
                placeholder="আপোনাৰ অসমীয়া পাঠ ইয়াত দিয়ক...",
                lines=4,
            )
            btn = gr.Button("Analyze Sentiment", variant="primary")

            gr.Examples(
                examples=[
                    ["এই নাটকখন সঁচাকৈয়ে ধুনীয়া আছিল, অভিনয়বোৰো বৰ ভাল লাগিল।"],
                    ["সময় নষ্ট। কাহিনীটো একেবাৰে বিৰক্তিকৰ।"],
                    ["চিনেমাখন সঁচাকৈয়ে বৰ ধুনীয়া হৈছে। পৰিয়ালৰ সৈতে চাবলৈ এখন সুন্দৰ চিনেমা।"],
                ],
                inputs=input_box,
            )

        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=2, label="Sentiment Confidence")
            output_details = gr.Markdown()

    gr.Markdown(MODEL_STATS)

    btn.click(fn=predict_sentiment, inputs=input_box, outputs=[output_label, output_details, gr.Textbox(visible=False)])
    input_box.submit(fn=predict_sentiment, inputs=input_box, outputs=[output_label, output_details, gr.Textbox(visible=False)])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
