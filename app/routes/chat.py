from fastapi import APIRouter
from app.services.ai_doctor import (
    doctor_response, extract_features,
    update_features, get_missing,
    next_question, FEATURES, RETRY_COUNT,
    reset_system
)
from app.services.explanation_service import rag_answer
import json

router = APIRouter()

# Use global session variables to track state
session_features = FEATURES.copy()
last_feat = None
rag_mode = False
awaiting_query = False

@router.post("/chat")
def chat(user_input: str | None = None):
    global session_features, last_feat, rag_mode, awaiting_query

    # 0. Handle initial greeting if no input
    if not user_input and not last_feat:
        return {"reply": "Hi there! 😊 I’m here to help you with your health and nutrition. I’ll ask you a few quick questions—just answer honestly, and I’ll guide you better. Type 'start' whenever you're ready!"}

    # 1. Handling RAG Mode Logic
    if rag_mode:
        if not user_input:
            return {"reply": "Do you have any health-related queries? (yes/no)"}
        
        text = user_input.lower().strip()
        
        # If we are waiting for the actual query text
        if awaiting_query:
            # Create a summary of patient data for the LLM to refer to
            patient_context = f"Patient Features: {json.dumps(session_features)}. "
            answer = rag_answer(user_input, patient_context=patient_context)
            awaiting_query = False
            return {
                "reply": f"{answer}\n\nDo you have any other query? (yes/no)",
                "rag_mode": True
            }
        
        # Checking for yes/no to enter/exit query loop
        if text in ["yes", "yeah", "yup", "y"]:
            awaiting_query = True
            return {
                "reply": "Please tell me your query",
                "rag_mode": True
            }
        elif text in ["no", "nope", "n"]:
            # EXIT RAG MODE and RESET
            reset_system()
            session_features = FEATURES.copy()
            last_feat = None
            rag_mode = False
            awaiting_query = False
            if hasattr(chat, "skipped"): chat.skipped = set()
            
            return {
                "reply": "Alright 👍 Let's start a new health assessment. Hi there! 😊 I’m here to help you with your health and nutrition. I’ll ask you a few quick questions—just answer honestly, and I’ll guide you better. Type 'start' whenever you're ready!",
                "status": "reset"
            }
        else:
            return {
                "reply": "Please answer with yes or no. Do you have any health-related queries?",
                "rag_mode": True
            }

    # 2. Handling Symptom Collection Mode
    text = user_input.lower().strip() if user_input else ""
    start_signals = ["hi", "hello", "hii", "hey", "hlo", "ok", "okk", "yes", "start"]
    
    # If starting fresh or user sends a start signal
    if any(s == text for s in start_signals) and not last_feat:
        reply = doctor_response(user_input)
        missing = get_missing(session_features)
        if missing:
            last_feat = missing[0]
            return {"reply": f"{reply}\n\n{next_question(last_feat)}"}
        return {"reply": reply}

    # If user sends something unrelated while we are waiting for a start signal
    if not last_feat:
        return {"reply": "Please type 'start' so I can help you better 😊"}

    extracted, note = extract_features(user_input, last_question_feat=last_feat)
    
    if last_feat and last_feat not in extracted:
        # If the input was completely unrelated and didn't match regex/numbers
        return {
            "reply": f"Please answer the question so I can help you better 😊\n\n{next_question(last_feat, is_retry=True)}"
        }

    session_features = update_features(session_features, extracted)
    
    missing = get_missing(session_features)
    if missing:
        last_feat = missing[0]
        reply_prefix = f"{note}\n\n" if note else ""
        return {
            "reply": f"{reply_prefix}{doctor_response(user_input)}\n\n{next_question(last_feat)}"
        }

    # All symptoms collected -> Enter RAG Mode for next turn
    # Ensure keys match the requested format and values are correctly represented
    order = [
        "Skin_darkening", "Hair_growth", "Weight_gain", "Lifestyle_Risk",
        "Cycle_Irregular", "Fast_food", "Pimples", "Hair_loss",
        "Hb", "MCV", "MCH", "MCHC", "Weight", "BMI", "Waist", "Hip", "BMI_Category"
    ]
    completed = {k: session_features.get(k, "") for k in order}
    
    rag_mode = True
    awaiting_query = False
    
    return {
        "status": "complete",
        "features": completed,
        "reply": "Assessment complete! ✅\n\nDo you have any health-related queries? (yes/no)"
    }
