# Copyright 2022 Google LLC (Refactored for ATD-12K Evaluation)
import os
from typing import Sequence
from pathlib import Path

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
import numpy as np

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 1. CLEANED UP FLAGS
# Removed the required frame1, frame2, and output_frame flags.
# We only keep the model_path and alignment configurations.
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default='pretrained_models/film_net/Style/saved_model',
    help='The path of the TF2 saved model to use.')
_ALIGN = flags.DEFINE_integer(
    name='align', default=64, help='If >1, pad the input size...')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height', default=1, help='Number of patches along height...')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width', default=1, help='Number of patches along width...')

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    print(f"Loading FILM model from {_MODEL_PATH.value}...")
    
    # 2. LOAD MODEL ONCE OUTSIDE THE LOOP
    # This prevents TensorFlow from rebuilding the graph 2,000 times!
    interpolator = interpolator_lib.Interpolator(
        model_path=_MODEL_PATH.value,
        align=_ALIGN.value,
        block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

    # 3. SETUP DATASET PATH
    dataset_dir = Path("/content/datasets/test_2k_540p")
    print(f"Starting FILM inference on dataset: {dataset_dir}")

    # Batched time = 0.5 (We want the exact middle frame)
    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    count = 0

    # 4. THE MASTER LOOP
    for folder in dataset_dir.iterdir():
        if folder.is_dir():
            img1_path = folder / "frame1.png"
            img3_path = folder / "frame3.png"
            
            # Save output exactly where we want it!
            save_path = folder / "frame2_film.png"

            if img1_path.exists() and img3_path.exists():
                
                # --- FILM'S IMAGE PREP ---
                image_1 = util.read_image(str(img1_path))
                image_batch_1 = np.expand_dims(image_1, axis=0)

                image_2 = util.read_image(str(img3_path))
                image_batch_2 = np.expand_dims(image_2, axis=0)

                # --- STREAMLINED INFERENCE ---
                mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

                # --- SAVE IMAGE ---
                util.write_image(str(save_path), mid_frame)

                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} triplets...")

    print(f"Finished successfully! Generated {count} FILM frames.")

if __name__ == '__main__':
    app.run(main)