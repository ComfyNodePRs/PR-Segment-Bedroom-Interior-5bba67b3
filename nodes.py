import torch
import torchvision.transforms.v2 as T
from .color_codes import COLOR_CODES

class BedroomFurnitureMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold": ("INT", { "default": 10, "min": 0, "max": 255, "step": 1, }),
            }
        }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "mask"

    def execute(self, image, threshold):
        # Convert the image to RGB values (0-255)
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)

        # Define the bedroom furniture items we want to mask
        bedroom_items = [
            "bed", "cabinet", "chair", "mirror", "rug;carpet;carpeting", 
            "lamp", "curtain;drape;drapery;mantle;pall"
        ]

        # Create an empty mask
        mask = torch.zeros(temp.shape[0], temp.shape[1], temp.shape[2], dtype=torch.float32)

        # For each bedroom item, create a mask and combine
        for item in bedroom_items:
            if item in COLOR_CODES:
                color = COLOR_CODES[item]
                color_tensor = torch.tensor(color).view(1, 1, 1, 3)
                lower_bound = (color_tensor - threshold).clamp(min=0)
                upper_bound = (color_tensor + threshold).clamp(max=255)
                color_mask = (temp >= lower_bound) & (temp <= upper_bound)
                color_mask = color_mask.all(dim=-1)
                mask |= color_mask.float()

        return (mask,)

BEDROOM_CLASS_MAPPINGS = {
    "BedroomFurnitureMask": BedroomFurnitureMask,
}

BEDROOM_NAME_MAPPINGS = {
    "BedroomFurnitureMask": "🛏️ Bedroom Furniture Mask",
}