import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from main import set_up_vit
from clip_feature import CLIPFeatureBranch
from preprocessing import Preprocessing
from fusion_model import FusionModel
from branch import Branch

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CHECKPOINT_PATH = "checkpoint.pt"
OUTPUT_PATH = "predictions.json"
BATCH_SIZE = 32

def load_model() -> tuple[FusionModel, Preprocessing, str]:
    clip_vit, image_processor = set_up_vit()
    branches = [CLIPFeatureBranch(clip_vit, image_processor=image_processor)]
    model = FusionModel(branches=branches, num_classes=1)
    preprocessing = Preprocessing(branches=branches)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # load parameters of frequency branch here
    model.classifier_head.load_state_dict(checkpoint['head_state'])

    if torch.cuda.is_available(): 
        device = "cuda" 
    else:
        device = "cpu"
    model.to(device)
    model.eval()

    return model, preprocessing, device

def collect_image_paths(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.rglob('*')) if p.suffix.lower() in IMAGE_EXTENSIONS]

#For input images
@torch.no_grad()
def run_batch(model: FusionModel, preprocessing: Preprocessing, device, image_paths: list[Path]):
    predicted_res = []

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]

        samples = []
        valid_paths = []

        #preprocess manually bec might not fit dataloader requirements
        for path in batch_paths:
            try:
                img = Image.open(path)
                img = img.convert('RGB')
                samples.append(preprocessing.full_transform(img))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not samples:
            continue

        pixel_dict = {}
        for branch in model.branches:
            if not isinstance(branch, Branch):
                raise TypeError('Expected instance of Branch but got ', type(branch))
            branch_name = branch.get_name()
            pixel_dict[branch_name] = torch.stack(
                [sample[branch_name] for sample in samples]
            ).to(device)

        logits = model(pixel_dict)
        probs_fake = torch.sigmoid(logits)

        for i in range(probs_fake.size(0)):
            predicted_res.append({
                "image_path": str(valid_paths[i]),
                "pred": round(probs_fake[i].item(), 4),
            })
    
    return predicted_res

def main():
    #takes in via CLI image directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Directory of images to score")
    args = parser.parse_args()
    if not args:
        raise Exception('No image directory found')

    model, preprocessing, device = load_model()
    image_paths = collect_image_paths(Path(args.image_dir))

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Found {len(image_paths)} images. Running predictions...")
    results = run_batch(model, preprocessing, device, image_paths)

    with open("output.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Wrote {len(results)} predictions to json file")

if __name__ == "__main__":
    main()