from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = "fine_tuned_model"  # Path to your fine-tuned model folder
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Parse the request data
        data = request.json
        prompt = data.get("prompt", "")
        max_length = data.get("max_length", 50)

        # Generate text using the model
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

        # Decode and send the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
