import json
from app.rag.rag_service import rag_service
from app.services.ai_doctor import ask_llm
from app.services.explain_service import get_model_explanations
from app.services.llm_service import generate_response, ask_hf_llm

def rag_answer(query: str, patient_context: str = "No patient data available yet.") -> str:
    """Handles direct Q&A after diagnosis using RAG context and local FLAN-T5."""
    try:
        # Retrieve context
        chunks = rag_service.query(query, k=3)
        if not chunks:
            return "👉 Sorry, currently I'm not able to give an answer. Please try again later."
        
        context_text = "\n\n".join(chunks)
        
        # Use the local generate_response
        res = generate_response(
            user_input=query,
            prediction_output=patient_context,
            rag_context=context_text
        )
        
        # Fallback Logic: If LLM output is too generic or short, use RAG chunks directly
        # flan-t5-small often struggles with natural generation, so direct extraction is safer.
        if not res or len(res.split()) < 10:
             # Combine the most relevant sentences from top chunks
             top_sentences = []
             for chunk in chunks[:2]:
                 # Filter out short header-like sentences
                 valid_sents = [s.strip() for s in chunk.split('.') if len(s.split()) > 4]
                 top_sentences.extend(valid_sents[:3])
             
             if top_sentences:
                 return " ".join(top_sentences) + "."
             return chunks[0]
             
        return res
    except Exception as e:
        print(f"RAG Q&A Error: {e}")
        return "👉 Sorry, currently I'm not able to give an answer. Please try again later."

def generate_explanation(anemia_res: dict, pcos_res: dict, features: dict) -> dict:
    """Generates a professional structured health analysis using RAG and local LLM."""
    # 1. Get XAI insights (SHAP values)
    xai_insights = get_model_explanations(features)
    
    anemia_factors = [f.split('(')[0].strip() for f in xai_insights.get('anemia', [])]
    pcos_factors = [f.split('(')[0].strip() for f in xai_insights.get('pcos', [])]
    
    # 2. Build RAG queries for context
    queries = []
    if anemia_res["prediction"] == 1: queries.extend(["Anemia symptoms and causes", "Iron rich diet"])
    if pcos_res["prediction"] == 1: queries.extend(["PCOS symptoms and lifestyle", "PCOS diet foods"])
    for feat in anemia_factors[:2] + pcos_factors[:2]:
        queries.append(f"Medical significance of {feat} in disease risk")
    
    # 3. Retrieve context
    context_chunks = []
    for q in queries[:6]:
        context_chunks.extend(rag_service.query(q, k=1))
    
    unique_context = list(set(context_chunks))
    context_text = "\n\n".join(unique_context)
    
    # 4. Prepare human-readable inputs
    label_map = {
        "Skin_darkening": "Skin Darkening", "Hair_growth": "Excess Hair Growth",
        "Weight_gain": "Recent Weight Gain", "Lifestyle_Risk": "Lifestyle Risks",
        "Cycle_Irregular": "Irregular Cycles", "Fast_food": "Junk Food Intake",
        "Pimples": "Acne/Pimples", "Hair_loss": "Hair Loss", "Hb": "Hemoglobin Level"
    }
    user_data_summary = ", ".join([f"{label_map.get(k, k)}: {v}" for k, v in features.items() if v not in [0, None, ""]])
    
    # Pre-synthesize XAI reasoning using RAG context for better sentences
    def get_context_sentence(feat, context_list):
        feat_low = feat.lower().replace('_', ' ')
        for chunk in context_list:
            if feat_low in chunk.lower():
                sentences = [s.strip() for s in chunk.split('.') if feat_low in s.lower() and len(s.split()) > 5]
                if sentences: return sentences[0] + "."
        return None

    anemia_insights = []
    for f in anemia_factors[:2]:
        ctx_s = get_context_sentence(f, unique_context)
        if ctx_s: anemia_insights.append(f"Regarding {f}: {ctx_s}")
    
    pcos_insights = []
    for f in pcos_factors[:2]:
        ctx_s = get_context_sentence(f, unique_context)
        if ctx_s: pcos_insights.append(f"Regarding {f.replace('_', ' ')}: {ctx_s}")

    # 5. Construct highly structured prompt with Markdown instructions
    full_prompt = f"""
    Instruction: You are a professional medical assistant. Generate a structured health analysis. 
    Use **Bold Headers**, bullet points, and proper spacing for readability.
    
    INPUT: 
    - Conditions: Anemia & PCOS Assessment 
    - Risks: Anemia ({anemia_res['risk']}), PCOS ({pcos_res['risk']})
    - Confidence: Anemia ({anemia_res['confidence']}), PCOS ({pcos_res['confidence']})
    - User Data: {user_data_summary} 
    - Medical Context: {context_text[:1200]} 
    
    OUTPUT FORMAT (MUST USE BOLD HEADERS):
    **1. Prediction Summary**
    - State condition, risk level, and confidence.
    
    **2. Key Contributing Factors**
    - List relevant factors from user data.
    
    **3. Model Insights (Explainability)**
    - Explain WHY the model flagged these risks. 
    - Use these pre-verified facts: {". ".join(anemia_insights + pcos_insights)}
    
    **4. Condition Overview**
    - Brief explanation from context.
    
    **5. Nutrition & Lifestyle**
    - Practical diet and habit tips.
    
    **6. Precautions**
    - When to see a doctor.
    
    Answer (Follow formatting strictly):
    """
    
    from app.services.llm_service import tokenizer, model
    import torch
    
    print("\n--- DEBUG: Generating Organized Health Analysis ---")
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=800, do_sample=True, temperature=0.3, repetition_penalty=1.2)
    
    analysis_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 6. Structured Fallback if LLM output is poor or messy
    if "**1." not in analysis_text or len(analysis_text.split()) < 50:
        analysis_text = f"""
**1. Prediction Summary**
- **Predicted Conditions:** Anemia & PCOS Assessment
- **Anemia Risk:** {anemia_res['risk']} ({anemia_res['confidence']})
- **PCOS Risk:** {pcos_res['risk']} ({pcos_res['confidence']})

**2. Key Contributing Factors**
- **Reported Data:** {user_data_summary}

**3. Model Insights (Explainability)**
- **Anemia Drivers:** {". ".join(anemia_insights) if anemia_insights else "Primary drivers include Hb and MCV levels."}
- **PCOS Drivers:** {". ".join(pcos_insights) if pcos_insights else "Primary drivers include cycle regularity and physical symptoms."}

**4. Condition Overview**
- {context_chunks[0] if context_chunks else "Based on medical context, these conditions are linked to hormonal and nutritional imbalances."}

**5. Nutrition & Lifestyle Recommendations**
- **Diet:** Focus on iron-rich foods (leafy greens, legumes) and insulin-regulating low-GI foods.
- **Lifestyle:** Maintain regular physical activity (150 min/week) and manage stress through meditation or yoga.

**6. Precautions / Alerts**
- Consult a healthcare professional if you experience severe fatigue, persistently irregular cycles, or sudden physical changes.

**Source Attribution:** Analysis based on trusted medical guidelines and RAG knowledge base.
"""

    return {
        "explanation": analysis_text,
        "dietary_recommendations": ["Refer to Section 5 above"],
        "lifestyle_recommendations": ["Refer to Section 5 above"]
    }
