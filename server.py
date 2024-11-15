from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
model_path = "fine_tuned_model"  # Path to your fine-tuned model folder
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set the pad token id to be the same as the eos_token_id (for GPT-2)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Parse the request data
        data = request.json
        prompt = data.get("prompt", "")
        max_length = data.get("max_length", 200)

        # Generate text using the model
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, device=inputs.device) 
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

        # Decode and send the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
