# Text-Summarization-App

## Prerequisite

Abstractive summarization is a technique for creating a summary of a text based on its primary ideas rather than by directly reproducing its most important sentences. This work in natural language processing is crucial and difficult. For this project, the **"Text to Text Transfer Transformer Model"** popularly known as T5 transformers was used to train our [custom dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary) to enable it give us the abstractive summary of the text data. T5 transformers was used because it is flexible enough to be fine-tuned for quite a number of important tasks especially **Abstractive Summarization**. T5 has also achieved state-of-the-art result in this field.
The framework used for this project is pytorch lightning and it was used because of its speed, efficiency, and reproducibility properties.

This project aims to sumamrize long text of 512 tokens or lesser, to tokens <= 128 without reproducing the words in the main text and also retaining context. The project was deployed using hugging face spaces with streamlit and this repo also contains a flask app which can be setup locally.

## Project Walkthrough.

**References**
<li>https://huggingface.co/docs/transformers/model_doc/t5</li>
<li>https://www.youtube.com/watch?v=KMyZUIraHio</li>
<li>https://www.sabrepc.com/blog/Deep-Learning-and-AI/why-use-pytorch-lightning</li>
<li>https://www.pytorchlightning.ai/</li>
