from transformers import pipeline
import time
import pysrt
# --- Configuration ---
MODEL_NAME = "google/flan-t5-base" 


def run_flan_t5_local(prompt: str):
    """
    Downloads the specified FLAN-T5 model and runs local inference for text generation.
    
    NOTE: The first time this script runs, it will download the model
    files (approx. 900MB) to your local cache. Subsequent runs will be fast.
    """
    print("--- 1. Model Loading (Download on first run) ---")
    print(f"Loading text generation model: {MODEL_NAME}")
    start_time = time.time()
    
    try:
        # The 'text2text-generation' pipeline is correct for T5 models.
        generator = pipeline(
            "text2text-generation", 
            model=MODEL_NAME
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds.")
        
        # --- 2. Local Inference ---
        print("-" * 50)
        print(f"Input Prompt: {prompt}")
        print("-" * 50)
        
        # Call the generator with the prompt
        results = generator(
            prompt, 
            max_length=50,       # Sets the maximum length of the generated response
            do_sample=True,      # Enables creative/varied responses
            temperature=0.7,      # Controls creativity (0.0 is deterministic, 1.0 is highly creative), with creativity comes hallucinations, I recommend not changing this value
            max_new_tokens=1000 
        )
        
        # --- 3. Display Result ---
        if results and results[0].get('generated_text'):
            generated_text = results[0]['generated_text']
            return generated_text
        else:
            print("Could not retrieve generated text.")

    except ImportError:
        print("Error: PyTorch or the transformers library is not installed.")
        print("Please run: pip install transformers torch")
    except Exception as e:
        print(f"An error occurred during model loading or generation: {e}")



def titler(srt_path:str = "titletester.srt"):
    subs = pysrt.open(srt_path, encoding='utf-8')
    full_text = " ".join(sub.text_without_tags.replace("\n", " ") for sub in subs)
    name = run_flan_t5_local(full_text)
    return name
