from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load local model and tokenizer explicitly to avoid pipeline task errors
# This resolves "RuntimeError: The model ... does not seem to have a correct pipeline_tag"
model_name = "google/flan-t5-small"
print(f"Loading local model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print(f"✅ Local model loaded successfully.")

def generate_response(user_input, prediction_output="", rag_context=""):
    """
    Generates a concise medical summary using a local FLAN-T5-Small model.
    """
    # Use a much more direct extractive prompt for the small model
    prompt = f"Context: {rag_context[:1000]}\n\nQuestion: {user_input}\n\nBased on the context, give a detailed answer (5-8 lines) about causes, symptoms and diet:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=450, # Increased for more detail
            do_sample=True, 
            temperature=0.6,
            repetition_penalty=1.3, # Higher penalty to avoid repetitive headers
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Validation: If response is too short or looks like a header, return a fallback
    if len(response.split()) < 10:
        return "" # Let the service layer handle the fallback
    
    return response

def ask_hf_llm(prompt: str, max_new_tokens: int = 300, system_message: str = None) -> str:
    """
    Compatibility wrapper for the previous ask_hf_llm function.
    Now redirects to the local generator with a simple prompt to avoid hallucinations.
    """
    # Simple, non-templated generation for conversational acknowledgments
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50, 
            do_sample=True, 
            temperature=0.3 # Low temperature for less randomness in acknowledgments
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
