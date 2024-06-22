from fastapi import FastAPI, Body, Depends
from typing import Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=['*'], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

def load_model():
    peft_model_id = "ANWAR101/lora-bart-base-youtube-cnn"
    config = PeftConfig.from_pretrained(peft_model_id) 
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    return model , tokenizer


@app.post("/summarize")
async def summarize(data: Dict[str, str] = Body(...)):
    """Summarize a text using the loaded Peft model."""
    model , tokenizer = load_model()

    text = data.get("text")

    # Check for missing text
    if not text:
        return {"error": "Missing text in request body"}, 400

    # Preprocess the text
    inputs = tokenizer(text, truncation=True, return_tensors="pt")

    # Generate summary using the model
    outputs = model.generate(
        **inputs, max_length=300, min_length=50, do_sample=True, num_beams=3,
        no_repeat_ngram_size=2, temperature=0.6, length_penalty=1.0
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = {"summary": summary}
    return response