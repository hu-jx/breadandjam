from paths import WILDFAKE_ZIPS_DIR
from modelscope.hub.api import HubApi
api = HubApi()
DALLE_ZIP_PATH = "Images/Diffusion_based/DALLE.zip"
COCO_ZIP_PATH = "Images/Real/coco.zip"
from modelscope.hub.snapshot_download import dataset_snapshot_download
import zipfile, os 

def extract_matching(zip_path, folder_name, out_dir, flatten=True):
    zf = zipfile.ZipFile(zip_path)
    members = [
        info for info in zf.infolist()
        if (f"/{folder_name}/" in info.filename or info.filename.startswith(f"{folder_name}/"))
        and not info.is_dir()          # <-- reliable directory check
    ]
    os.makedirs(out_dir, exist_ok=True)
    for info in members:
        data = zf.read(info.filename)
        target = os.path.join(out_dir, info.filename.replace("/", "__")) if flatten else os.path.join(out_dir, info.filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
    print(f"Extracted {len(members)} files matching folder '{folder_name}' from {zip_path}")

def load_wildfake_data(real_dir:str, fake_dir: str):
    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        return
    dataset_snapshot_download(
        dataset_id="hy2628982280/WildFake",
        allow_patterns=["Images/Real/coco.zip",
            "Images/Diffusion_based/DALLE.zip",],
        local_dir=WILDFAKE_ZIPS_DIR
    )   

    extract_matching(
        f"{WILDFAKE_ZIPS_DIR}/Images/Real/coco.zip",
        "val2017",
        real_dir
    )

    extract_matching(
        f"{WILDFAKE_ZIPS_DIR}/Images/Diffusion_based/DALLE.zip",
        "Advanced",
        fake_dir
    )






