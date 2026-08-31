from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

def set_up_vit():
    clip_vit = AutoModelForZeroShotImageClassification.from_pretrained(
        "openai/clip-vit-large-patch14",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    image_processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    return clip_vit, image_processor