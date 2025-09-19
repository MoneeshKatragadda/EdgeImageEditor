# src/edit_utils.py (Upgraded to be "Mask-Aware")

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class FastEditor:
    """
    A class for applying fast, deterministic image edits.
    All methods now accept an optional 'mask' argument to apply edits locally.
    """
    def __init__(self):
        print("FastEditor initialized.")

    def _apply_masked_edit(self, original_image: Image.Image, edited_image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Helper function to blend an edited image with an original using a mask."""
        # Convert the boolean mask from SAM into a PIL-compatible format
        mask_pil = Image.fromarray(mask).convert("L")
        # Composite the images together. Where the mask is "white", the edited_image is used.
        # Where the mask is "black", the original_image is used.
        return Image.composite(edited_image, original_image, mask_pil)

    def adjust_brightness(self, image: Image.Image, factor: float, mask: np.ndarray = None) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        edited_image = enhancer.enhance(factor)
        if mask is not None:
            return self._apply_masked_edit(image, edited_image, mask)
        return edited_image

    def adjust_contrast(self, image: Image.Image, factor: float, mask: np.ndarray = None) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        edited_image = enhancer.enhance(factor)
        if mask is not None:
            return self._apply_masked_edit(image, edited_image, mask)
        return edited_image

    def adjust_saturation(self, image: Image.Image, factor: float, mask: np.ndarray = None) -> Image.Image:
        enhancer = ImageEnhance.Color(image)
        edited_image = enhancer.enhance(factor)
        if mask is not None:
            return self._apply_masked_edit(image, edited_image, mask)
        return edited_image

    def apply_blur(self, image: Image.Image, radius: int = 2, mask: np.ndarray = None) -> Image.Image:
        edited_image = image.filter(ImageFilter.GaussianBlur(radius))
        if mask is not None:
            return self._apply_masked_edit(image, edited_image, mask)
        return edited_image

    # Geometric transformations (rotate, resize) are typically global and don't use masks
    def rotate(self, image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
        return image.rotate(angle, expand=expand)

    def resize(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        return image.resize(size, Image.Resampling.LANCZOS)