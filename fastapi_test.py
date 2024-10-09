from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize FastAPI app
app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your model's name
model = AutoModelForCausalLM.from_pretrained("gpt2")  # Replace with your model's name

# Define the generate endpoint
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()  # Parse incoming JSON request body
    prompt = data.get("message")  # Extract 'message' from the parsed JSON

    if not prompt:
        return {"error": "No 'message' found in request."}

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    # Decode generated output into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}  # Return the generated text as response

# To run the FastAPI app, use uvicorn:
# uvicorn main:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
