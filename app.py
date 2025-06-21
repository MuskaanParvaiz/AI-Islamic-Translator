import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# --- Load Model & Tokenizer ---
@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model = PeftModel.from_pretrained(base_model, "./nllb-quran-peft")
    tokenizer = AutoTokenizer.from_pretrained("./nllb-quran-peft")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

model, tokenizer = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Arabic to Hindi Translator", layout="centered")
st.title("🕌 Arabic to Hindi Quranic Translator")
st.markdown("Enter Quranic Arabic text to get a Hindi translation using your fine-tuned NLLB model.")

# Input box
arabic_input = st.text_area("✍️ Enter Arabic Text Here:")

# Translate button
if st.button("🌐 Translate"):
    if not arabic_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Translating..."):
            input_text = ">>hin_Deva<< " + arabic_input
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=150)
            translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success("✅ Hindi Translation:")
            st.markdown(f"**{translated_text}**")
