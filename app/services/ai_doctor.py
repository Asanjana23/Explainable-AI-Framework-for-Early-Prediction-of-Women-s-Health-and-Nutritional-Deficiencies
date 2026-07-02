import os
import re
import json
import random
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from app.services.llm_service import ask_hf_llm

load_dotenv()

chat_memory: List[Dict[str, str]] = []
MAX_MEMORY = 10

FEATURES: Dict[str, Any] = {
    "Hb": None, "MCV": None, "MCH": None, "MCHC": None,
    "Gender": 0,
    "Skin_darkening": None, "Hair_growth": None, "Weight_gain": None,
    "Lifestyle_Risk": None, "Cycle_Irregular": None, "Fast_food": None,
    "Pimples": None, "Weight": None, "BMI": None, "Hair_loss": None,
    "Waist": None, "Hip": None, "BMI_Category": None
}

RETRY_COUNT: Dict[str, int] = {}
rag_mode = False

def reset_system():
    global chat_memory, FEATURES, RETRY_COUNT, rag_mode
    chat_memory = []
    # Re-initialize FEATURES with None values
    for k in FEATURES:
        if k != "Gender":
            FEATURES[k] = None
    RETRY_COUNT = {}
    rag_mode = False

def _clip_memory():
    global chat_memory
    if len(chat_memory) > MAX_MEMORY:
        chat_memory = chat_memory[-MAX_MEMORY:]

def normalize_input(value):
    if isinstance(value, str):
        v = value.strip().lower()
        
        # Explicit YES
        if v in ["yes", "y", "yeah", "yep", "true", "1"]:
            return 1, None
            
        # Interpretive YES
        if v in ["little bit", "some", "sometimes", "a bit", "occasionally"]:
            return 1, "Thanks for sharing that 👍 I’ll consider that as a 'yes'."
            
        # Explicit NO
        if v in ["no", "n", "nope", "not really", "rarely", "never", "false", "0"]:
            return 0, None
            
        if v in ["normal"]:
            return 0, None
        if v in ["overweight"]:
            return 1, None
        if v in ["obese"]:
            return 2, None
            
        try:
            # Handle cases like "hb is 12"
            num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", v)
            if num_match:
                return float(num_match.group(1)), None
            return float(v), None
        except Exception:
            return value, None
    return value, None

def ask_llm(prompt: str, fallback_msg: Optional[str] = None) -> str:
    """Hugging Face Mistral-7B-Instruct implementation."""
    response = ask_hf_llm(prompt)
    if response and response not in ["MODEL_LOADING", "Sorry, I couldn't respond right now."]:
        return response
    
    if fallback_msg:
        return fallback_msg
    
    return "Sorry, I couldn't respond right now."

def doctor_response(user_input: str) -> str:
    """Rule-based chat response layer (Only use LLM for final diagnosis)."""
    global chat_memory
    
    text = user_input.lower().strip()
    start_signals = ["hi", "hello", "hii", "hey", "hlo", "ok", "okk", "yes", "start"]
    
    # 1. Start Signal Logic
    if any(s == text for s in start_signals):
        reply = "Great! Let’s get started 👍"
    
    # 2. Numeric/Simple Data Logic (Avoid LLM for data entry)
    # This prevents hallucinations like "The woman was having a difficult time..."
    elif re.search(r"^[0-9]+(?:\.[0-9]+)?$", text) or text in ["normal", "overweight", "obese", "yes", "no"]:
        acknowledgments = [
            "Got it 👍", "Noted.", "Okay, thanks.", 
            "Recorded.", "Thanks for that.", "Understood."
        ]
        reply = random.choice(acknowledgments)
    
    # 3. Conversational Fallback
    else:
        # Use LLM ONLY for actual conversational replies, with a very simple prompt
        # We bypass the heavy medical template for simple chat
        prompt = f"Give a very short one-line medical acknowledgment to: '{user_input}'"
        reply = ask_hf_llm(prompt)
        
        # Safe fallback if LLM still acts up
        if not reply or len(reply.split()) > 10:
             reply = "Got it, I've recorded that. Let's move to the next question."

    chat_memory.append({"role": "user", "text": user_input})
    chat_memory.append({"role": "assistant", "text": reply})
    _clip_memory()
    return reply

def extract_features(user_input: str, last_question_feat: Optional[str] = None) -> (Dict[str, Any], Optional[str]):
    """Purely rule-based (Regex) feature extraction with flexible normalization."""
    text = user_input.lower().strip()
    out: Dict[str, Any] = {}
    note: Optional[str] = None

    # Boolean feature mapping
    bool_map = ["Skin_darkening", "Hair_growth", "Weight_gain", "Lifestyle_Risk", 
                "Cycle_Irregular", "Fast_food", "Pimples", "Hair_loss"]

    # 1. Direct answer logic (if we know what we asked)
    if last_question_feat:
        val, msg = normalize_input(text)
        
        if last_question_feat in bool_map:
            if val in [0, 1]:
                out[last_question_feat] = val
                note = msg
                return out, note
        
        numeric_feats = ["Hb", "MCV", "MCH", "MCHC", "Weight", "BMI", "Waist", "Hip"]
        if last_question_feat in numeric_feats:
            if isinstance(val, (int, float)):
                out[last_question_feat] = val
                return out, None
        
        if last_question_feat == "BMI_Category":
            if val in [0, 1, 2]:
                out[last_question_feat] = val
                return out, None

    return out, None

def update_features(current: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new_data.items():
        if k in current and k != "Gender":
            if v is not None:
                current[k] = v
    return current

def get_missing(features: Dict[str, Any]):
    return [k for k, v in features.items() if v is None]

def next_question(feature: str, is_retry: bool = False) -> str:
    global RETRY_COUNT
    if not is_retry:
        RETRY_COUNT[feature] = RETRY_COUNT.get(feature, 0) + 1
    
    questions = {
        "Skin_darkening": "Do you have skin darkening (Hyperpigmentation)? (yes/no)",
        "Hair_growth": "Do you have excess hair growth (Hirsutism)? (yes/no)",
        "Weight_gain": "Have you experienced weight gain recently? (yes/no)",
        "Lifestyle_Risk": "Do you have lifestyle risk factors (e.g., lack of exercise, stress, poor diet)? (yes/no)",
        "Cycle_Irregular": "Are your menstrual cycles irregular (Irregular Periods)? (yes/no)",
        "Fast_food": "Do you eat fast food frequently (Junk Food Consumption)? (yes/no)",
        "Pimples": "Do you have pimples (Acne)? (yes/no)",
        "Hair_loss": "Do you experience hair loss (Hair Fall)? (yes/no)",
        "Hb": "What is your Hemoglobin (Hb) level? (g/dL)",
        "MCV": "What is your Mean Corpuscular Volume (MCV)? (fL)",
        "MCH": "What is your Mean Corpuscular Hemoglobin (MCH)? (pg)",
        "MCHC": "What is your Mean Corpuscular Hemoglobin Concentration (MCHC)? (g/dL)",
        "Weight": "What is your body weight (kg)?",
        "BMI": "What is your Body Mass Index (BMI)?",
        "Waist": "What is your waist size (cm)?",
        "Hip": "What is your hip size (cm)?",
        "BMI_Category": "What is your BMI category (Body Mass Index Category)? (normal/overweight/obese)"
    }
    
    q = questions.get(feature, f"Please provide {feature}")
    
    if is_retry:
        if feature in ["Skin_darkening", "Hair_growth", "Weight_gain", "Lifestyle_Risk", "Cycle_Irregular", "Fast_food", "Pimples", "Hair_loss"]:
            return f"Please answer yes or no. {q}"
        elif feature in ["Hb", "MCV", "MCH", "MCHC", "Weight", "BMI", "Waist", "Hip"]:
            return f"Please provide a valid number. {q}"
        elif feature == "BMI_Category":
            return f"Please select from: normal, overweight, or obese. {q}"
            
    return q
