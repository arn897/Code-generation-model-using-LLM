Code Generation Model Using LLM
A deep learning project that fine-tunes a Large Language Model (LLM) for generating Python code snippets based on user prompts. This project demonstrates how to train, fine-tune, and deploy a code generation model using the Hugging Face Transformers library and PyTorch.

Table of Content:
Project Overview
Features
Installation
Usage
Model Training
Generating Code
Demo
Technologies Used
Contributing
License



Project Overview
This project fine-tunes a transformer-based Large Language Model (CodeGen) to generate Python code snippets. Given a user prompt describing a function or algorithm, the model produces relevant and syntactically correct Python code. The model is trained on custom datasets and can generate code for a variety of algorithms.

Features
Fine-tuning of pre-trained CodeGen model

Generation of Python functions based on text prompts

Support for custom prompts and max generation length

GPU acceleration for faster training and inference

Sample scripts for testing and evaluation

Installation
Make sure you have Python 3.8+ installed.

Clone the repo:

bash
Copy
Edit
git clone https://github.com/arn897/Code-generation-model-using-LLM.git
cd Code-generation-model-using-LLM
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Loading the fine-tuned model and generating code
python
Copy
Edit
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./results/checkpoint-438"  # Path to fine-tuned checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_code(prompt, max_length=350):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip() if generated.startswith(prompt) else generated

prompt = "def quick_sort(arr):"
print(generate_code(prompt))
Model Training
To fine-tune the model on your custom dataset, use the training scripts with Hugging Face's Trainer API. Ensure you have GPU support enabled.

Demo
You can interactively generate code by running:

python
Copy
Edit
prompt = input("Enter your code prompt: ")
print(generate_code(prompt))
Technologies Used
Python 3.8+

PyTorch

Hugging Face Transformers

CUDA (for GPU acceleration)

Kaggle (for notebook environment)

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

