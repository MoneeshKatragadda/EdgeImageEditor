# src/rl_environment.py (Final Version with Structured Data)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import os
from PIL import Image
import torch

# Import our powerful, custom-built tools
from src.edit_utils import FastEditor
from src.segment_utils import ObjectSegmenter
from src.clip_utils import CLIPScorer

class ImageEditEnv(gym.Env):
    """A custom Gymnasium environment for object-level, prompt-driven image editing."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, image_paths: list, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # --- NEW: Define a structured training dataset ---
        # This explicitly lists the valid, segmentable objects for each image.
        self.training_data = [
            {"path": "data/test_images/Car.jpg", "objects": ["car", "mountain", "sky", "road", "headlight", "tire"]},
            {"path": "data/test_images/elephant.jpg", "objects": ["elephant", "sun", "water", "sky"]},
            {"path": "data/test_images/Lake.jpg", "objects": ["lake", "mountain", "sky", "forest", "person"]},
            {"path": "data/test_images/People.jpg", "objects": ["people", "person"]},
            {"path": "data/test_images/City.jpg", "objects": ["building", "road", "sky"]},
            {"path": "data/test_images/Temple.jpg", "objects": ["temple", "statue", "sky"]},
            {"path": "data/test_images/Church.jpg", "objects": ["church", "tree", "sky", "clock"]},
            {"path": "data/test_images/Bike.jpg", "objects": ["bike", "person", "helmet"]},
            {"path": "data/test_images/Park.jpeg", "objects": ["slide", "trees", "swing set"]},
            {"path": "data/test_images/Anime_building.jpeg", "objects": ["building", "road"]},
            {"path": "data/test_images/Anime_tree.jpeg", "objects": ["trees"]}
        ]
        
        # We also need a dictionary of aesthetic prompts for each object type
        self.training_prompts = {
            "car": ["a photo of a car with high contrast", "a cinematic photo of a car"],
            "sky": ["a photo of a vibrant blue sky", "a dramatic, moody sky with deep colors"],
            "mountain": ["a photo of a majestic, sharp mountain", "a soft, dreamy mountain landscape"],
            "road": ["a photo of a dark, wet road", "a bright, clear road"],
            "headlight": ["a bright, glowing headlight"], "tire": ["a clean, black tire"],
            "elephant": ["a majestic elephant in a dramatic sunset", "a sharp, high-contrast photo of an elephant"],
            "sun": ["a bright, glowing sun"], "water": ["a photo of calm, reflective water"],
            "lake": ["a photo of a vibrant, turquoise lake", "a calm, serene lake with soft reflections"],
            "person": ["a portrait with soft lighting", "a vibrant photo of a person with colorful clothes"],
            "people": ["a happy group of people"],
            "building": ["a sharp, detailed photo of a building", "a warm, inviting building at sunset"],
            "temple": ["a photo of a golden, glowing temple", "a detailed and awe-inspiring temple"],
            "statue": ["a dramatic photo of a statue with strong shadows", "a bright, marble-white statue"],
            "church": ["a classic, sharp photo of a church", "a moody, atmospheric photo of a church"],
            "clock": ["a detailed clock face"],
            "bike": ["a gritty, high-contrast photo of a motorcycle", "a cinematic shot of a bike in nature"],
            "helmet": ["a shiny, clean helmet"],
            "tree": ["a photo of a tree with vibrant green leaves", "a dark and moody tree silhouette"],
            "trees": ["a photo of a tree with vibrant green leaves", "a dark and moody tree silhouette"],
            "forest": ["a photo of a lush, green forest", "a dark, mystical forest with deep shadows"],
            "slide": ["a colorful, fun slide"], "swing set": ["a playful swing set"]
        }
        
        # Initialize tools
        self.editor = FastEditor()
        self.segmenter = ObjectSegmenter(device=self.device)
        self.clip_scorer = CLIPScorer(device=self.device)
        
        self.episode_count = 0
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.action_map = { 0: self.editor.adjust_brightness, 1: self.editor.adjust_contrast, 2: self.editor.adjust_saturation }
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.8], dtype=np.float32),
            high=np.array([float(len(self.action_map)), 1.2], dtype=np.float32),
            shape=(2,)
        )
        
        self.obs_size = 512
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_size, self.obs_size, 4), dtype=np.uint8
        )
        
        self.current_image_pil = None
        self.target_mask = None
        self.target_prompt = None
        self.previous_score = 0.0
        self.current_step = 0
        self.max_steps = 8

    def _save_debug_image(self, image_cv2, mask, box, noun):
        overlay = image_cv2.copy()
        color = (30, 144, 255)
        alpha = 0.5
        overlay[mask] = cv2.addWeighted(overlay[mask], alpha, np.full(overlay[mask].shape, color, dtype=np.uint8), 1 - alpha, 0)[0]
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"Detected: '{noun}'"
        cv2.putText(overlay, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        save_path = os.path.join(self.debug_dir, f"episode_{self.episode_count}.jpg")
        cv2.imwrite(save_path, overlay)
        print(f"--- Saved debug image to: {save_path} ---")

    def _get_initial_state(self):
        mask = None
        while mask is None:
            # --- MODIFIED LOGIC ---
            # 1. Pick a random image entry from our structured data
            image_data = random.choice(self.training_data)
            image_path = image_data["path"]
            
            # 2. Pick a random object from the list of valid objects for that image
            target_object_noun = random.choice(image_data["objects"])
            
            print(f"\n--- Attempting to start new episode. Target: '{target_object_noun}' in image '{os.path.basename(image_path)}' ---")
            image_cv2 = cv2.imread(image_path)
            if image_cv2 is None:
                print(f"Warning: Failed to load image {image_path}. Skipping.")
                continue
            
            mask, box = self.segmenter.segment(image_cv2, target_object_noun)

            if mask is None:
                print(f"Could not find '{target_object_noun}'. This can happen, retrying...")
        
        self.episode_count += 1
        self._save_debug_image(image_cv2, mask, box, target_object_noun)
        
        self.target_prompt = random.choice(self.training_prompts[target_object_noun])
        self.current_image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        self.current_image_pil = self.current_image_pil.resize((self.obs_size, self.obs_size), Image.Resampling.LANCZOS)
        
        mask_pil = Image.fromarray(mask)
        self.target_mask = np.array(mask_pil.resize((self.obs_size, self.obs_size), Image.Resampling.NEAREST))
        
        print(f"--- Successfully started new episode. Goal: '{self.target_prompt}' ---")
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._get_initial_state()
        self.previous_score = self.clip_scorer.score(self.current_image_pil, self.target_prompt)
        self.current_step = 0
        observation = np.dstack((np.array(self.current_image_pil), (self.target_mask > 0).astype(np.uint8) * 255))
        return observation, {"prompt": self.target_prompt, "object": "unknown"}

    def step(self, action):
        action_type_float, amount = action
        action_type = int(round(action_type_float))
        action_type = max(0, min(action_type, len(self.action_map) - 1))
        amount = float(amount)
        edit_function = self.action_map.get(action_type)
        if edit_function:
            self.current_image_pil = edit_function(self.current_image_pil, amount, mask=self.target_mask)
        new_score = self.clip_scorer.score(self.current_image_pil, self.target_prompt)
        reward = new_score - self.previous_score
        self.previous_score = new_score
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        observation = np.dstack((np.array(self.current_image_pil), (self.target_mask > 0).astype(np.uint8) * 255))
        info = {"score": new_score}
        return observation, reward, done, truncated, info

    def render(self):
        return np.array(self.current_image_pil)

    def close(self):
        pass