# demo.py (Final Interactive Version)

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import all our custom tools
from src.segment_utils import ObjectSegmenter
from src.edit_utils import FastEditor

def run_demo():
    # --- 1. Configuration ---
    # Use the final, object-aware model you just trained
    MODEL_PATH = "models\ppo_image_editor_final_10000_steps.zip" 
    # Or whatever your latest saved model is named, e.g., ppo_image_editor_final_150000_steps.zip
    
    TEST_IMAGE_PATH = "data/test_images/Car.jpg" # The image you want to edit
    OBS_SIZE = 512
    MAX_STEPS = 8
    
    # --- 2. Initialize all components ---
    print("--- Initializing AI components ---")
    segmenter = ObjectSegmenter(device="cuda" if torch.cuda.is_available() else "cpu")
    editor = FastEditor()
    model = PPO.load(MODEL_PATH)
    print("All components loaded successfully.")

    # --- 3. Get User Input ---
    print("\n--- Welcome to the AI Image Editor ---")
    print(f"Editing image: {os.path.basename(TEST_IMAGE_PATH)}")
    
    # Get the editing prompt from the user
    user_prompt = input("Enter your editing prompt (e.g., 'make the car high contrast'): ")
    if not user_prompt:
        print("No prompt given. Exiting.")
        return

    # --- 4. Segment the Object ---
    # We need to parse the noun from the prompt for the segmenter
    noun_prompt, _ = segmenter._parse_prompt(user_prompt)
    print(f"\n1. Finding '{noun_prompt}' in the image...")
    
    original_image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
    
    mask, box = segmenter.segment(image_cv2, noun_prompt)
    
    if mask is None:
        print(f"Sorry, I could not find '{noun_prompt}' in the image. Please try another prompt.")
        return
    print(f"Object '{noun_prompt}' found successfully.")

    # --- 5. Prepare for Inference ---
    # Resize image and mask to the agent's observation size
    observation_image_pil = original_image_pil.resize((OBS_SIZE, OBS_SIZE), Image.Resampling.LANCZOS)
    mask_pil = Image.fromarray(mask).resize((OBS_SIZE, OBS_SIZE), Image.Resampling.NEAREST)
    target_mask = np.array(mask_pil)

    # Create the 4-channel observation (RGB + Mask)
    observation = np.dstack((
        np.array(observation_image_pil),
        (target_mask > 0).astype(np.uint8) * 255
    ))

    # Keep a full-resolution copy for applying edits
    edited_full_res_image_pil = original_image_pil.copy()

    # --- 6. Run the Inference Loop ---
    print("\n2. Applying learned edits from the AI agent...")
    for step in range(MAX_STEPS):
        action, _states = model.predict(observation, deterministic=True)
        
        # Interpret the agent's chosen action
        action_type = int(round(action[0]))
        amount = float(action[1])

        # Define a helper to apply the edit
        def apply_edit(image_to_edit, action_type, amount, mask_to_use):
            edit_map = { 0: editor.adjust_brightness, 1: editor.adjust_contrast, 2: editor.adjust_saturation }
            edit_function = edit_map.get(action_type)
            if edit_function:
                print(f"  Step {step+1}: Applying {edit_function.__name__} with factor {amount:.2f}")
                return edit_function(image_to_edit, amount, mask=mask_to_use)
            return image_to_edit

        # Apply edit to the full-res image using the original full-res mask
        edited_full_res_image_pil = apply_edit(edited_full_res_image_pil, action_type, amount, mask)
        
        # Apply edit to the observation image for the agent's next step
        observation_image_pil = apply_edit(observation_image_pil, action_type, amount, target_mask)
        observation = np.dstack((np.array(observation_image_pil), (target_mask > 0).astype(np.uint8) * 255))

    print("--- Editing complete! ---")

    # --- 7. Display the Results ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(original_image_pil)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(edited_full_res_image_pil)
    ax[1].set_title(f"Final Edit (Prompt: '{user_prompt}')")
    ax[1].axis('off')
    plt.show()

if __name__ == '__main__':
    # Need to add this to use torch in the main script
    import torch
    run_demo()