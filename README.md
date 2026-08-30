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

## Deployment

Deployed on [Render](https://render.com) (free tier). Build command: `pip install -r requirements.txt`. Start command: `python app.py`.

**Note:** the free tier sleeps after 15 minutes of inactivity — the first request after a period of inactivity may take 30-60 seconds to respond while the server wakes up.

## Related

- 🔬 [Full project (5-model comparison)](https://github.com/AvinabhDutta-Dev/assamese-sentiment-analysis)
- 📊 [Dataset](https://huggingface.co/datasets/AvinabhDutta-Dev/assamese-movie-reviews-sentiment)
- 🤗 [Model](https://huggingface.co/AvinabhDutta-Dev/assamese-sentiment-muril)
