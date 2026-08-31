# Assamese Sentiment Analysis — Live Demo

A live, interactive Gradio app for Assamese sentiment classification, powered by a fine-tuned MuRIL model (96.33% accuracy). Deployed on Render for free public hosting, embedded directly in my portfolio.

## Model

Uses [AvinabhDutta-Dev/assamese-sentiment-muril](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril), one of five models benchmarked in a comparative study on an original, dual-annotated 8,986-review Assamese movie review dataset.

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

## Model

This demo uses an ONNX-converted version of the fine-tuned MuRIL model, hosted on Hugging Face:

**[AvinabhDutta-Dev/assamese-sentiment-muril-onnx](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril-onnx)**

The ONNX format was chosen specifically to reduce memory footprint for lightweight CPU deployment — the original, fully documented PyTorch model (with complete evaluation results and model card) is available separately at [AvinabhDutta-Dev/assamese-sentiment-muril](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril).

The model is loaded directly from Hugging Face Hub at runtime — no model files are stored in this repository.

## Related

- 🔬 [Full project (5-model comparison)](https://github.com/AvinabhDutta-Dev/assamese-sentiment-analysis)
- 📊 [Dataset](https://huggingface.co/datasets/AvinabhDutta-Dev/assamese-movie-reviews-sentiment)
- 🤗 [Model](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril)
- 🤗 [Model](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril-onnx)
