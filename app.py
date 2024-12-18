from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
model_path = "fine_tuned_model"  # Path to your fine-tuned model folder
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad_token_id to eos_token_id, since GPT-2 does not have a pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Store password securely as an environment variable (set it in your server environment)
PASSWORD = os.getenv('PASSWORD', 'ThisIsAPassword!!')  # Default password for testing

@app.route('/')
def home():
    return "Flask app is running! Use the /generate endpoint."

@app.route('/generate', methods=['POST'])
def generate():
     try:
        # Parse the request data
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request, JSON data is required"}), 400

        prompt = data.get("prompt", "")
        password = data.get("password", "")

        # Check the password
        if password != PASSWORD:
            return jsonify({"error": "Unauthorized access: Incorrect password"}), 403

        # Validate the prompt
        if not prompt.strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Generate text using the model
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, 
            max_length=200, 
            num_return_sequences=1, 
            attention_mask=torch.ones(inputs.shape, device=inputs.device)
        )

        # Decode and send the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"generated_text": generated_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
