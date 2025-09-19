# test_env.py (Updated for the final environment)

from stable_baselines3.common.env_checker import check_env
from src.rl_environment import ImageEditEnv
import os

print("--- Testing the FINAL custom ImageEditEnv ---")

# Create a dummy list of image paths for the test
image_folder = "data/test_images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg'))]

if not image_paths:
    raise ValueError("No images found in data/test_images. Please add at least one.")

# 1. Instantiate the new environment
# Note: It no longer needs a target_prompt at the start
env = ImageEditEnv(image_paths=image_paths, device="cpu")

# 2. Run the environment checker
try:
    check_env(env)
    print("\n✅ Final Environment check passed!")
    print("Your object-aware environment is valid and ready for the final training.")
except Exception as e:
    print(f"\n❌ Environment check failed: {e}")