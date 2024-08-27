import torch
from .color_codes import COLOR_CODES

class BedroomFurnitureMask:
    FURNITURE_ITEMS = [
        "bed", "cabinet", "chair", "mirror", "rug;carpet;carpeting",
        "lamp", "curtain;drape;drapery;mantle;pall", "painting;picture","pillow",
        "shelf", "cushion", "tree", "vase", "base;pedestal;stand", "flower",
        "book", "box", "blanket;cover", "ottoman;pouf;pouffe;puff;hassock"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
            }
        }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "mask"

    def execute(self, image, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        com_mask = torch.zeros(temp.shape[0], temp.shape[1], temp.shape[2], dtype=torch.bool)
        for item in self.FURNITURE_ITEMS:
            if item in COLOR_CODES:
                color = COLOR_CODES[item]
                red = color[0]
                green = color[1]
                blue = color[2]
                color = torch.tensor([red, green, blue])
                lower_bound = (color - threshold).clamp(min=0)
                upper_bound = (color + threshold).clamp(max=255)
                lower_bound = lower_bound.view(1, 1, 1, 3)
                upper_bound = upper_bound.view(1, 1, 1, 3)
                mask = (temp >= lower_bound) & (temp <= upper_bound)
                mask = mask.all(dim=-1)
                com_mask |= mask
            else:
                print(f"Warning: Color code not found for '{item}'. Skipping.")

        return (com_mask.float(),)

BEDROOM_CLASS_MAPPINGS = {
    "BedroomFurnitureMask": BedroomFurnitureMask,
}

BEDROOM_NAME_MAPPINGS = {
    "BedroomFurnitureMask": "🛏️ Bedroom Furniture Mask",
}