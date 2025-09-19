# train.py (Final, ROBUST Version with Checkpointing and Resume Logic)

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.rl_environment import ImageEditEnv
import glob
import torch

# --- 1. Configuration ---
TOTAL_TIMESTEPS = 150_000
STEPS_PER_SAVE = 10_000 # How often to save a checkpoint
MODEL_NAME = "ppo_image_editor_final"

# --- 2. Setup ---
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

image_folder = "data/test_images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

if not image_paths:
    raise ValueError(f"No images found in {image_folder}. Please add training images.")

# --- 3. Create the Environment ---
print("--- Creating the Environment ---")
env = ImageEditEnv(
    image_paths=image_paths, 
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("Environment created.")

# --- 4. Setup Checkpoint Callback ---
checkpoint_callback = CheckpointCallback(
  save_freq=STEPS_PER_SAVE,
  save_path=model_dir,
  name_prefix=MODEL_NAME
)

# --- 5. Find the Latest Checkpoint to Resume From (if any) ---
model = None
list_of_files = glob.glob(os.path.join(model_dir, f"{MODEL_NAME}_*.zip"))
if list_of_files:
    # --- THIS IS THE CORRECTED LINE ---
    latest_file = max(list_of_files, key=lambda f: int(f.split('_')[-2]))
    
    print(f"--- Resuming training from checkpoint: {os.path.basename(latest_file)} ---")
    model = PPO.load(latest_file, env=env, tensorboard_log=log_dir)
else:
    print("--- No checkpoint found. Starting a new training run. ---")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

# --- 6. Train the Agent ---
print(f"--- Starting/Resuming Training for {TOTAL_TIMESTEPS} Timesteps ---")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    reset_num_timesteps= (not list_of_files) # Don't reset if we are resuming
)

print("--- Training Complete ---")

# --- 7. Save the Final Agent ---
final_model_path = os.path.join(model_dir, f"{MODEL_NAME}_final")
model.save(final_model_path)
print(f"✅ Final Model saved to {final_model_path}.zip")

# --- 8. Clean up ---
env.close()