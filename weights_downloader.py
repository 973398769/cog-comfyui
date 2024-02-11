import subprocess
import time
import os

from weights_manifest import WeightsManifest

BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"


class WeightsDownloader:
    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/fofr/cog-comfyui/blob/main/weights_licenses.md"
                )
            self.download_if_not_exists(
                weight_str,
                self.weights_map[weight_str]["url"],
                self.weights_map[weight_str]["dest"],
            )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/fofr/cog-comfyui/blob/main/supported_weights.md"
            )

    def download_torch_checkpoints(self):
        self.download_if_not_exists(
            "mobilenet_v2-b0353104.pth",
            f"{BASE_URL}/custom_nodes/comfyui_controlnet_aux/mobilenet_v2-b0353104.pth.tar",
            "/root/.cache/torch/hub/checkpoints/",
        )

    def download_if_not_exists(self, weight_str, url, dest):
        if not os.path.exists(f"{dest}/{weight_str}"):
            self.download(weight_str, url, dest)

    def download(self, weight_str, url, dest):
        if weight_str == "swizz8_REALBakedvaeFP16.safetensors":
            print("weight_str is swizz8, reset download url")
            url = "https://civitai.com/api/download/models/180074"
        if weight_str == "bbox/hand_yolov8s.pt":
            print("weight_str is hand yolo8s, reset download url")
            url = "https://huggingface.co/Bingsu/adetailer/blob/main/hand_yolov8s.pt"
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        if weight_str == "swizz8_REALBakedvaeFP16.safetensors":
            dest = "ComfyUI/models/checkpoints/swizz8_REALBakedvaeFP16.safetensors"
            print("weight_str is swizz8/handYolo, reset download command")
            subprocess.check_call(
                # ["wget", "-o", "swizz8_REALBakedvaeFP16.safetensors", "https://civitai.com/api/download/models/180074", "-P", dest], close_fds=False
                ["pget", "--log-level", "warn", "-f", url, dest], close_fds=False
            )
        elif weight_str == "bbox/hand_yolov8s.pt":
            print("weight_str is handYolo, reset download command")
            dest = "ComfyUI/models/ultralytics/bbox/hand_yolov8s.pt"
            subprocess.check_call(
                # ["wget", "-o", "hand_yolov8s.pt", "https://civitai.com/api/download/models/180074",
                #  "-P", dest], close_fds=False
                ["pget", "--log-level", "warn", "-f", url, dest], close_fds=False
            )
        else:
            subprocess.check_call(
                ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
            )

        elapsed_time = time.time() - start
        try:
            dest_file_path = os.path.join(dest, os.path.basename(weight_str))
            print(f"The weight is {dest_file_path}")
            if os.path.exists(dest_file_path):
                file_size_bytes = os.path.getsize(
                    os.path.join(dest_file_path)
                )
                file_size_megabytes = file_size_bytes / (1024 * 1024)
                print(
                    f"⌛️ Downloaded {weight_str} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
                )
            else:
                print(f"The {dest_file_path} is not exist")
        except FileNotFoundError:
            print(f"Warning: Could not get the file size for {weight_str}")
