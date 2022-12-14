# Text-Summarization-App

## Introduction

- Abstractive summarization is a technique for creating a summary of a text based on its primary ideas rather than by directly reproducing its most important sentences. This work in natural language processing is crucial and difficult. For this project, the **"Text to Text Transfer Transformer Model"** popularly known as T5 transformers was used to train our [custom dataset](https://www.kaggle.com/datasets/sunnysai12345/news-summary) to enable it give us the abstractive summary of the text data. 
- T5 transformers was used because it is flexible enough to be fine-tuned for quite a number of important tasks especially **Abstractive Summarization**. T5 has also achieved state-of-the-art result in this field.
- The framework used for this project is pytorch lightning and it was used because of its speed, efficiency, and reproducibility properties.

This project aims to sumamrize long text of 512 tokens or lesser, to tokens <= 128 without reproducing the words in the main text and also retaining context. The project was deployed using hugging face spaces with streamlit and this repo also contains a flask app which can be setup locally.

### Repository Structure 
**Project files/folders:**
  <li>Static: This folder contains the css file for the U.I of the flask app</li>
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
<li>Summary: This function is used to generate summary on a piece of text by the user. This is the function that is used along with the fine-tuned model for deployment.</li>
  
 ## Project Walkthrough.
 The section aims to give a walkthrough of the [model training](https://github.com/hemhemoh/Text-Summarization-App/blob/main/T5_transformers.ipynb) and the deployment aspect of the project.
 ### Text preprocessing and Model building
 
 - Setup and library imports: This is the first phase of any Machine learning data and it is at this point we switch to GPU(if necessary), install and import the libraries we need to get the project started(e.g numpy, torch, pytorch lightning). Some of these libraries can be imported/installed later in the project but I prefer to install mine at the beginning of my ipynb notebook.
 - Getting the dataset from kaggle to google colab: Instead of manualing downloading the dataset this was done by using the kaggle library and the API command of the dataset on kaggle.
 - Loading the dataset into a dataframe.
 - Examining the dataframe: The dataframe consists of 6 columns, 4 of which are unnecessary for this task. A new dataframe containing just the text and summary columns was created. 
 - Data cleaning: The column names were changed to relevant names, rows containing null data was dropped as these rows are just few and filling them would be hard.
 - Splitting the dataset into test and train data: The new dataframe created was splitted into train and test column with test_size of 0.1.
 - NewsSummaryDataset: This class was created to encode, [tokenize](https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4#:~:text=Tokenization%20is%20breaking%20the%20raw,the%20sequence%20of%20the%20words), pad and truncate the dataset.This is done with the t5 tokenizer from the T5 model installed.This class also specify the maximum token length for both the text column and the summary column. 
 - NewsSummaryDataModule: This class applies the function above to our dataset and takes into cognizance the train test split done above.
 - NewsSummaryModel: This class contains functions that trains and tests the model.
 - A checkpoint was created for the model and a trainer which was created from our pytorch lightning module was used to fit the data and the model.
 - A function summarize was created which was used to test the model built. This function takes in the text to be summarized and returns the summary of length <= 128.

  
## Limitations
<p> The T5-base model is a very large one and can't be deployed on free cloud platforms because of the size.</p>

## Blockers
I found it hard building my flask APP with vscode, I am not sure why but I guess dependency was a major issue. Setting it on Pycharm was less challenging and I had to install RUST, CYTHON and one or two other dependencies in the git bash terminal.

**References**
<li>https://huggingface.co/docs/transformers/model_doc/t5</li>
<li>https://www.youtube.com/watch?v=KMyZUIraHio</li>
<li>https://www.sabrepc.com/blog/Deep-Learning-and-AI/why-use-pytorch-lightning</li>
<li>https://www.pytorchlightning.ai/</li>
