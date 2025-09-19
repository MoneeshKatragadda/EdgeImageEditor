# src/clip_utils.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    """
    A class to handle loading the CLIP model and calculating scores.
    """
    def __init__(self, device="cpu"):
        self.device = device
        model_id = "openai/clip-vit-base-patch32"
        
        # Load the pre-trained CLIP model and its processor from Hugging Face
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print("CLIPScorer initialized on device:", self.device)

    def score(self, image: Image.Image, text_prompt: str) -> float:
        """
        Calculates the similarity score between a given image and a text prompt.
        
        Returns:
            A float between 0 and 1 representing the similarity.
        """
        # Prepare the image and text for the model
        inputs = self.processor(
            text=[text_prompt], 
            images=[image], 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Get the model's output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate the cosine similarity and convert it to a regular Python float
        # The logits_per_image is the similarity score
        similarity = outputs.logits_per_image.item()
        
        # The raw score can be around ~20-30. We can scale it to be nicer.
        # This scaling is optional but helps keep rewards in a smaller range.
        scaled_score = similarity / 100.0
        
        return scaled_score