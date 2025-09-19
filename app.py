# ai_prompt_refiner_v2_deployable.py
import streamlit as st
from ctransformers import AutoModelForCausalLM
import re

# --- Page Config ---
st.set_page_config(
    page_title="AI Prompt Refiner",
    layout="wide"
)

# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner="Loading the AI Prompt Engineer (TinyLlama)...")
def load_llm():
    """
    Loads the TinyLlama model, which is small enough for Streamlit's free tier.
    """
    # --- MODEL FIXED FOR DEPLOYMENT ---
    return AutoModelForCausalLM.from_pretrained(
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama", # Changed model type
        gpu_layers=0,
        context_length=2048 # Adjusted for TinyLlama
    )

# --- Core Logic: The Master Prompt ---
def refine_prompt(llm, initial_prompt, context):
    """
    Uses the LLM with a master prompt to refine the user's initial prompt.
    """
    system_prompt = """
    You are 'PromptPerfect', an expert AI prompt engineer. Your task is to take a user's simple prompt and rewrite it into three distinct, high-quality, and detailed versions to get the best possible response from a powerful AI model.

    For each refined prompt, apply a combination of these techniques:
    - **Persona:** Assign a role to the AI (e.g., "Act as a world-class chef...").
    - **Format:** Specify the desired output format (e.g., "Provide the output in a JSON object...").
    - **Context:** Incorporate the user's provided context.
    - **Examples (Few-shot):** Provide a clear example of the desired output.
    - **Constraints:** Set rules or negative constraints (e.g., "Do not use technical jargon.").
    - **Chain of Thought:** Instruct the AI to "think step-by-step".

    Structure your response as follows, using "---" as a separator:

    ### Refined Prompt 1
    **Techniques Used:** [List the techniques you applied]
    ```
    (Your first refined prompt goes here)
    ```
    ---
    ### Refined Prompt 2
    **Techniques Used:** [List the techniques you applied]
    ```
    (Your second refined prompt goes here)
    ```
    """
    
    user_content = f"Initial Prompt: \"{initial_prompt}\"\n\nOptional Context: \"{context}\""

    # --- PROMPT FORMAT UPDATED FOR TINYLLAMA ---
    full_prompt = f"<|im_start|>user\n{system_prompt}\n\nHere is the user's request:\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        response = llm(full_prompt, max_new_tokens=1500, temperature=0.8)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# --- Streamlit UI ---
st.title("🚀 AI Prompt Refiner")
st.caption("Transform your simple ideas into powerful, high-quality prompts for any AI.")

llm = load_llm()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Idea")
    initial_prompt = st.text_area("Enter your basic prompt:", height=100, placeholder="e.g., explain black holes")
    context = st.text_area("Optional: Add some context:", height=200, placeholder="e.g., explain it to a 10-year-old child")

    if st.button("✨ Refine My Prompt", use_container_width=True):
        if not initial_prompt:
            st.warning("Please enter a prompt to refine.")
        else:
            with st.spinner("The AI Prompt Engineer is thinking..."):
                refined_output = refine_prompt(llm, initial_prompt, context)
                st.session_state.refined_output = refined_output

with col2:
    st.subheader("Refined Prompts")
    if 'refined_output' in st.session_state and st.session_state.refined_output:
        # Split the output into individual prompt sections
        prompts = st.session_state.refined_output.split('---')
        for i, p in enumerate(prompts):
            if p.strip():
                st.markdown(p.strip())

st.info("This app runs a lightweight open-source model perfect for deployment. The first run will be a bit slow as the model is downloaded.")
