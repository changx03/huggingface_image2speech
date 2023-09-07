# Huggingface🤗 Image To Speech Demo

Tested and updated on: `8-Sep-2023`.

Based on AI Jason's YouTube tutorial ([Original YouTube link](https://youtu.be/_j7JEDWuqLE?si=0vQ_yitVcOu7wXgY))

This example creates a web-app so user can upload an image, and the pre-trained model generate a story based on the image.

## OpenAI API

The LLM uses `ChatGPT3`. The maximum length is set to 50 words.

To use the OpenAI API, you need create a `.env` file with your `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY=<OPENAI_PRIVATE_KEY>
MAX_TOKENS=50
```

## To install packages

Tested on `Python=3.9.18`

From your virtual environment:

```bash
pip install -r requirements.txt
```

## To run the web server from localhost

```bash
streamlit run app.py
```
