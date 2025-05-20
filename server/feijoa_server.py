"""VGGT Server for Feijoa use-case."""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from server.vggt_inference import get_vggt_model, vggt_inference

app = FastAPI()

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)

# Load VGGT model
model = get_vggt_model()


@app.post("/vggt_predict/")
async def vggt_predict(images: list[UploadFile] = File(...)):
    # Setup directory for this VGGT run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vggt_dir = results_dir / timestamp
    image_dir = vggt_dir / "input_images"
    out_dir = vggt_dir / "outputs"
    out_images_dir = out_dir / "images"
    out_depth_dir = out_dir / "depth"
    for d in [vggt_dir, image_dir, out_dir, out_images_dir, out_depth_dir]:
        d.mkdir()
    print(f"VGGT results directory: {vggt_dir}")

    # Write input images
    image_paths, image_fnames = [], []
    for image in images:
        safe_name = Path(image.filename).name
        path = image_dir / safe_name
        with open(path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        image_paths.append(str(path))
        image_fnames.append(safe_name)

    # Run VGGT and extract predictions
    vggt_predictions = vggt_inference(model, image_paths)
    rgbs = vggt_predictions["images"][0].cpu().numpy()
    depth_map = vggt_predictions["depth"][0].cpu().numpy()
    intrinsics = vggt_predictions["intrinsic"][0].cpu().numpy()

    # Compute c2w
    w2cs = torch.eye(4)[None].repeat(len(images), 1, 1)
    w2cs[:, :3, :4] = vggt_predictions["extrinsic"][0]
    c2ws = torch.linalg.inv(w2cs).cpu().numpy()

    # Write outputs
    transforms = {"frames": []}
    for fname, rgb, depth, c2w, intrinsic in zip(
        image_fnames, rgbs, depth_map, c2ws, intrinsics
    ):
        # RGB
        rgb_img = (np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(
            str(out_images_dir / fname), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        )

        # Depth
        depth_path = (out_depth_dir / fname).with_suffix(".png")
        depth_uint16 = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_uint16)

        transforms["frames"].append(
            {
                "file_path": str((out_images_dir / fname).relative_to(out_dir)),
                "depth_file_path": str(depth_path.relative_to(out_dir)),
                "transform_matrix": c2w.tolist(),
                "intrinsic_matrix": intrinsic.tolist(),
            }
        )

    # Save depth confidence
    depth_conf = vggt_predictions["depth_conf"][0].cpu().numpy()
    depth_conf_path = out_dir / "depth_conf.npy"
    np.save(str(depth_conf_path), depth_conf)
    transforms["depth_confidence_path"] = str(depth_conf_path.relative_to(out_dir))

    # Save JSON
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Wrote results to {out_dir}")

    # Zip results
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in out_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(out_dir))
    print(f"Wrote zip to {zip_path}")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )


@app.get("/")
def read_root():
    return {"message": "VGGT inference server is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("feijoa_server:app", host="0.0.0.0", port=1234)
