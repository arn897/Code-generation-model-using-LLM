# 🚀 Code Generation Model using LLM (Fine-Tuned CodeGen)

Transforming natural language into working Python code using a fine-tuned Large Language Model (LLM). This project showcases how you can build a code generator that accepts prompts like _"Write a binary search function"_ and returns clean, executable Python code. 🧠💻

---

## 📌 Project Highlights

- 🤖 Fine-tuned `Salesforce/codegen-350M-multi` using Hugging Face Transformers  
- 🧑‍💻 Generates Python functions from user-written prompts  
- ⚡ GPU-accelerated training and inference  
- 📦 Works in notebooks or scripts for easy integration  
- 🔁 Uses sampling methods like `top_k`, `top_p` for diverse generations  

---

## 📂 Folder Structure

├── data/ # Your dataset
├── results/ # Checkpoints after training
├── CodeGenLLM.ipynb # Training and inference notebook
├── generate.py # Script to generate code from prompt
├── README.md
└── requirements.txt

---

## 🔧 Setup Instructions


# 1. Clone the repository
git clone https://github.com/arn897/Code-generation-model-using-llm.git
cd code-generation-llm

# 2. (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt


💡 How to Use
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./results/checkpoint-438"  # Path to saved model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

def generate_code(prompt, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
print(generate_code("def quick_sort(arr):"))

📊 Demo Output
Prompt:

python
Copy
Edit
def binary_search(arr, target):
Generated Output:

python
Copy
Edit
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

🧠 Model Info

Base Model: Salesforce/codegen-350M-multi

Trained On: Custom prompts + Python code samples

Checkpoint Used: checkpoint-438

Framework: PyTorch + Hugging Face Transformers

📌 Technologies

Python 🐍

Hugging Face 🤗

PyTorch 🔥

Jupyter Notebooks 📓

CUDA (for GPU acceleration) 🖥️


