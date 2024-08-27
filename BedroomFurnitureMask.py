import torch
from color_codes import COLOR_CODES

class BedroomFurnitureMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "image/masking"

    def __init__(self):
        self.bedroom_items = [
            "bed", "cabinet", "chair", "mirror", "rug;carpet;carpeting", 
            "lamp", "curtain;drape;drapery;mantle;pall"
        ]

    def create_mask(self, image, threshold):
        # Convert the image to RGB values (0-255)
        rgb_image = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)

        # Create an empty mask
        mask = torch.zeros(rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2], dtype=torch.float32)

        # For each bedroom item, create a mask and combine
        for item in self.bedroom_items:
            if item in COLOR_CODES:
                color = COLOR_CODES[item]
                color_tensor = torch.tensor(color).view(1, 1, 1, 3)
                color_mask = torch.all(torch.abs(rgb_image - color_tensor) <= threshold, dim=-1)
                mask |= color_mask

        return (mask,)

# Don't forget to add these lines at the end of your file
BEDROOM_CLASS_MAPPINGS = {
    "BedroomFurnitureMask": BedroomFurnitureMask
}

BEDROOM_DISPLAY_NAME_MAPPINGS = {
    "BedroomFurnitureMask": "Bedroom Furniture Mask"
}