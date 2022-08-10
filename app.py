import streamlit as st
import torch
from transformers import(
    T5TokenizerFast as T5Tokenizer)
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device('cpu')
model = torch.load('models.pth', map_location=device)


def summarize(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")

    generated_ids = model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True)

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]

    return "".join(preds)


def main():
    """Text Summarizer app with streamlit"""
    st.title("T5 text summarizer with streamlit")
    st.subheader("Summarize your 512 words here!")
    message = st.text_area("Enter your text", "Type Here")
    if st.button("Summarize text"):
        summary_results = summarize(message)
        st.write(summary_results)

if __name__ == '__main__':
    main()
