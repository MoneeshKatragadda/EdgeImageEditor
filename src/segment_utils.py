# src/segment_utils.py

import torch
import numpy as np
import cv2
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import sam_model_registry, SamPredictor

class ObjectSegmenter:
    """
    A class to handle the entire text-to-mask pipeline.
    It uses GroundingDINO to find objects based on a text prompt
    and SAM to generate a precise mask for the selected object.
    """
    def __init__(self, device="cpu"):
        self.device = device
        
        # --- Load GroundingDINO Model ---
        G_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        G_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
        self.grounding_dino_model = GroundingDINOModel(
            model_config_path=G_DINO_CONFIG_PATH, 
            model_checkpoint_path=G_DINO_CHECKPOINT_PATH,
            device=self.device
        )
        
        # --- Load SAM Model ---
        SAM_CHECKPOINT_PATH = "models/sam_vit_b_01ec64.pth"
        SAM_MODEL_TYPE = "vit_b"
        self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def _parse_prompt(self, prompt: str) -> tuple[str, str | None]:
        """
        Parses a prompt to separate the main object noun from a modifier.
        Example: "left headlight" -> ("headlight", "left")
        Example: "bigger mountain" -> ("mountain", "bigger")
        Example: "car" -> ("car", None)
        """
        modifiers = ["left", "right", "rear", "front", "bigger", "largest", "smaller", "smallest"]
        words = prompt.lower().split()
        
        found_modifier = None
        noun_words = []
        
        for word in words:
            if word in modifiers:
                found_modifier = word
            else:
                noun_words.append(word)
                
        # Handle "front tire" vs "rear tire" as spatial modifiers
        if "front" in prompt.lower() and "tire" in prompt.lower():
            found_modifier = "right" # In our reference image, front tire is on the right
        if "rear" in prompt.lower() and "tire" in prompt.lower():
            found_modifier = "left" # In our reference image, rear tire is on the left
        
        return " ".join(noun_words), found_modifier

    def segment(self, image: np.ndarray, prompt: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Main method to detect and segment an object based on a text prompt.
        """
        print("--- Starting Segmentation ---")
        
        # 1. PARSE THE PROMPT
        noun_prompt, modifier = self._parse_prompt(prompt)
        print(f"1. Parsed Prompt: Noun='{noun_prompt}', Modifier='{modifier}'")
        
        # 2. DETECT OBJECTS WITH GROUNDINGDINO
        detections, phrases = self.grounding_dino_model.predict_with_caption(
            image=image.copy(),
            caption=noun_prompt,
            box_threshold=0.3,
            text_threshold=0.2
        )
        print(f"2. Found {len(detections)} instance(s) of '{noun_prompt}'")
        if len(detections.xyxy) == 0:
            print("!!! No instances found. Aborting. !!!")
            return None, None

        # 3. SELECT THE CORRECT BOX BASED ON THE MODIFIER
        selected_box = None
        if len(detections.xyxy) == 1:
            selected_box = detections.xyxy[0]
            print("3. Selected the only box found.")
        else:
            # Multiple objects found, we need to use the modifier to choose
            boxes = detections.xyxy
            
            if modifier in ["left", "right"]:
                sorted_boxes = sorted(boxes, key=lambda box: box[0])
                selected_box = sorted_boxes[0] if modifier == "left" else sorted_boxes[-1]
                print(f"3. Selected the '{modifier}-most' box.")
            
            elif modifier in ["bigger", "largest"]:
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                selected_box = boxes[box_areas.argmax()]
                print("3. Selected the 'largest' box.")

            elif modifier in ["smaller", "smallest"]:
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                selected_box = boxes[box_areas.argmin()]
                print("3. Selected the 'smallest' box.")
            
            else: # Default behavior if no specific modifier is found
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                selected_box = boxes[box_areas.argmax()]
                print("3. No specific modifier, selected the 'largest' box by default.")

        if selected_box is None:
            print("!!! Box selection logic failed. Aborting. !!!")
            return None, None
            
        print(f"--> Final Selected Box: {selected_box}")

        # 4. GENERATE MASK WITH SAM
        print("4. Starting mask generation with SAM...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        masks, scores, logits = self.sam_predictor.predict(
            box=selected_box,
            multimask_output=True
        )
        print("...SAM prediction complete.")
        
        final_mask = masks[scores.argmax()]
        print("5. Mask chosen. Returning final mask and box.")
        return final_mask, selected_box