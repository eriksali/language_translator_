# pip install gradio torch transformers

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def translate_text(text):
    inputs = tokenizer.encode("translate English to French: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="Speech Translation")

generator = gr.Interface.load("huggingface/facebook/wav2vec2-base-960h",
                              inputs="microphone",
                              outputs=output_1,
                              title="Speech-to-text",
                          )

translator = gr.Interface(fn=translate_text,
                          inputs=output_1,
                          outputs=output_2,
                          title="English to French Translator",
                          description="Translate English speech to French text using the T5-small model.",
                          )

gr.Series(generator, translator).launch()
