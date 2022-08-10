from flask import Flask, render_template, request
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer, AdamW
import warnings
warnings.filterwarnings("ignore")


# class NewsSummaryDataset(Dataset):
#
#     def __inpython main.py
#     hit__(
#             self,
#             data: pd.DataFrame,
#             tokenizer: T5Tokenizer,
#             text_max_token_len: int = 512,
#             summary_max_token_len: int = 128
#     ):
#         self.tokenizer = tokenizer
#         self.data = data
#         self.text_max_token_len = text_max_token_len
#         self.summary_max_token_len = summary_max_token_len
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index: int):
#         data_row = self.data.iloc[index]
#
#         text = data_row["text"]
#
#         text_encoding = tokenizer(
#             text,
#             max_length=self.text_max_token_len,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt"
#         )
#
#         summary_encoding = tokenizer(
#             data_row['summary'],
#             max_length=self.summary_max_token_len,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt"
#         )
#
#         labels = summary_encoding["input_ids"]
#         labels[labels == 0] = -100
#
#         return dict(
#             text=text,
#             summary=data_row["summary"],
#             text_input_ids=text_encoding["input_ids"].flatten(),
#             text_attention_mask=text_encoding["attention_mask"].flatten(),
#             labels=labels.flatten(),
#             labels_attention_mask=summary_encoding["attention_mask"].flatten()
#         )
#
#
# class NewsSummaryDataModule(pl.LightningDataModule):
#
#     def __init__(
#             self,
#             train_df: pd.DataFrame,
#             test_df: pd.DataFrame,
#             tokenizer: T5Tokenizer,
#             batch_size: int = 8,
#             text_max_token_len: int = 512,
#             summary_max_token_len: int = 128
#     ):
#         super().__init__()
#
#         self.train_df = train_df
#         self.test_df = test_df
#
#         self.batch_size = batch_size
#         self.tokenizer = tokenizer
#         self.text_max_token_len = text_max_token_len
#         self.summary_max_token_len = summary_max_token_len
#
#     def setup(self, stage=None):
#         self.train_dataset = NewsSummaryDataset(
#             self.train_df,
#             self.tokenizer,
#             self.text_max_token_len,
#             self.summary_max_token_len
#         )
#         self.test_dataset = NewsSummaryDataset(
#             self.test_df,
#             self.tokenizer,
#             self.text_max_token_len,
#             self.summary_max_token_len
#         )
#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=2
#         )
#     def val_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=2
#         )
#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=2
#         )
#
#
# class NewsSummaryModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
#
#     def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
#         output = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             decoder_attention_mask=decoder_attention_mask
#         )
#         return output.loss, output.logits
#
#     def training_step(self, batch, batch_idx):
#         input_ids = batch["text_input_ids"]
#         attention_mask = batch["text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels
#         )
#
#         self.log("train_loss", loss, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         input_ids = batch["text_input_ids"]
#         attention_mask = batch["text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels
#         )
#
#         self.log("val_loss", loss)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         input_ids = batch["text_input_ids"]
#         attention_mask = batch["text_attention_mask"]
#         labels = batch["labels"]
#         labels_attention_mask = batch["labels_attention_mask"]
#
#         loss, outputs = self(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             decoder_attention_mask=labels_attention_mask,
#             labels=labels
#         )
#
#         self.log("test_loss", loss, prog_bar=True, logger=True)
#         return loss
#
#     def configure_optimizers(self):
#         return AdamW(self.parameters(), lr=0.0001)


def summarize(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    generated_ids = model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]

    return "".join(preds)


MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

device = torch.device('cpu')
model = torch.load('models.pth', map_location=device)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        summary = summarize(data)
    return render_template('result.html', my_summary=summary)


if __name__ == '__main__':
    app.run()
