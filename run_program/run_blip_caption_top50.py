"""
Generate BLIP captions for a subset of prediction records.

Reads `predictions_with_captions.jsonl`, takes the first 50 entries,
produces `output/xai_gpt/predictions_with_captions_blip_top50.jsonl`
with field `image_caption` filled using BLIP.

Usage:
    python run_program/run_blip_caption_top50.py

Dependencies:
    - transformers (BLIP)
    - pillow
    - tqdm
    - torch (CPU wheel is fine)
"""

import json
from itertools import islice
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


BASE = Path("output/tombert_ultra_optimized_20251129_225613_20251129_225622")
PRED_PATH = BASE / "predictions_with_captions.jsonl"
IMAGES_DIR = Path("absa_data/twitter2015_images")
OUT_DIR = Path("output/xai_gpt")
OUT_PATH = OUT_DIR / "predictions_with_captions_blip_top50.jsonl"


def caption_image(img_path: Path, processor: BlipProcessor, model: BlipForConditionalGeneration) -> str | None:
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=30)
    return processor.decode(out[0], skip_special_tokens=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_safetensors=True
    )

    with PRED_PATH.open() as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in tqdm(islice(f_in, 50), total=50, desc="BLIP captioning (50)", unit="img"):
            rec = json.loads(line)
            img_id = str(rec.get("image_id", "")).strip()
            cap = None
            if img_id:
                img_file = IMAGES_DIR / img_id
                if not img_file.exists() and not img_id.lower().endswith(".jpg"):
                    alt = IMAGES_DIR / f"{img_id}.jpg"
                    if alt.exists():
                        img_file = alt
                if img_file.exists():
                    cap = caption_image(img_file, processor, model)
            rec["image_caption"] = cap or f"(caption tidak tersedia, image_id={img_id})"
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

