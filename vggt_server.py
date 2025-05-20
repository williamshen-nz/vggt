"""Modified from https://github.com/facebookresearch/vggt/blob/main/demo_gradio.py"""

import time
from functools import lru_cache

import uvicorn
import logging
import os
import shutil
import uuid

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


_log = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please run on a machine with a GPU.")


@lru_cache(maxsize=1)
def get_model():
    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)
    print(f"VGGT model loaded successfully to {device}")
    return model


@app.post("/predict/")
async def predict(images: list[UploadFile] = File(...)):
    # Write images to unique target directory
    target_id = uuid.uuid4().hex
    target_dir = f"/tmp/input_images_{target_id}"
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
    print(f"Writing images to {target_dir}")

    image_paths = []
    for img in images:
        img_path = os.path.join(target_dir, "images", img.filename)
        with open(img_path, "wb") as f:
            shutil.copyfileobj(img.file, f)
        image_paths.append(img_path)

    # Load and preprocess images
    torch.cuda.empty_cache()
    img_tensor = load_and_preprocess_images(sorted(image_paths)).to(device)
    print(f"Loaded images to tensor of shape {img_tensor.shape}")

    # Forward pass
    model = get_model()
    print("Running inference...")
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        predictions = model(img_tensor)
    forward_dur = time.perf_counter() - start_time
    print(f"Inference completed in {forward_dur:.2f}s")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], img_tensor.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0).tolist()

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = np.array(predictions["depth"])
    world_points = unproject_depth_map_to_point_map(
        depth_map,
        np.array(predictions["extrinsic"]),
        np.array(predictions["intrinsic"]),
    )
    predictions["world_points_from_depth"] = world_points.tolist()

    # Clean up
    torch.cuda.empty_cache()
    # Just return success
    return JSONResponse(
        content={
            "message": "Inference completed successfully.",
        },
        status_code=200,
    )
    # return JSONResponse(content=predictions)


# Optional health check
@app.get("/")
def read_root():
    return {"message": "VGGT inference server is running."}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("vggt_server:app", host="0.0.0.0", port=1234, reload=True)
