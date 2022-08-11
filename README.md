# Text-Summarization-App

## Prerequisite

- Abstractive summarization is a technique for creating a summary of a text based on its primary ideas rather than by directly reproducing its most important sentences. This work in natural language processing is crucial and difficult. For this project, the **"Text to Text Transfer Transformer Model"** popularly known as T5 transformers was used to train our [custom dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary) to enable it give us the abstractive summary of the text data. 
- T5 transformers was used because it is flexible enough to be fine-tuned for quite a number of important tasks especially **Abstractive Summarization**. T5 has also achieved state-of-the-art result in this field.
- The framework used for this project is pytorch lightning and it was used because of its speed, efficiency, and reproducibility properties.

This project aims to sumamrize long text of 512 tokens or lesser, to tokens <= 128 without reproducing the words in the main text and also retaining context. The project was deployed using hugging face spaces with streamlit and this repo also contains a flask app which can be setup locally.

## Project Walkthrough.

### File Folder Structure 
**Project files/folders:**
  <li>Static: This folder contains the css file for the U.I of the flask app<li>
  <li>Template: This folder contains the HTML files for the home and predict page of the flask app</li>
  <li>T5 transformers.ipynb: This is the google colab notebook used for preprocessing the data and fine tuning the model.</li>
  <li>App.py: This is the streamlit file created for the purpose of being the U.I file for the model deployed on hugging face spaces.</li>
  <li>Main.py:This is the flask file created for the deployment of the model on cloud platforms.</li>

### Dataset
This project was started by getting a dataset that could work with the T5 transformer model since the model takes in texts and returns text(one of the main reason it is flexible). Data preprocessing and exploratory data analysis was carried out on the dataset before going ahead to use the tokenizer on the model.
##Add graph photo here##


**Pytorch-lightning** is the framework used in this project and that's majorly because it helps organize our pytorch code.
Some of the functions created using pytorch lightning includes:
<li>NewsSummaryDataset which was used to tokenize and encode the dataset</li>
<li>NewsSummaryDataModule which ws used to setup the output from the NewsSummaryDataset function and also to load them into dataloaders</li>
<li>NewsSummaryModel: this is where the version of T5(t5-base) to be used was specified and downloaded. The model architecture was alsp fine tuned for custom dataset here.</li>
<li>Summary: This function is used to generate summary on a piece of text by the user. This is the function that is used along with the fine-tuned model for deployment.<li>
  

  
## Limitations
<p> The T5-base model is a very large one and cant be deployed on free cloud platforms because of the size.</p>

**References**
<li>https://huggingface.co/docs/transformers/model_doc/t5</li>
<li>https://www.youtube.com/watch?v=KMyZUIraHio</li>
<li>https://www.sabrepc.com/blog/Deep-Learning-and-AI/why-use-pytorch-lightning</li>
<li>https://www.pytorchlightning.ai/</li>
