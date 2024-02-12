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
            url = "https://cdn-lfs.huggingface.co/repos/0d/db/0ddb8d3fcb6ee9737d9dd7e090e6c6cb6e40728d12307180160e7f654cb345de/30878cea9870964d4a238339e9dcff002078bbbaa1a058b07e11c167f67eca1c?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27hand_yolov8s.pt%3B+filename%3D%22hand_yolov8s.pt%22%3B&Expires=1707967321&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNzk2NzMyMX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy8wZC9kYi8wZGRiOGQzZmNiNmVlOTczN2Q5ZGQ3ZTA5MGU2YzZjYjZlNDA3MjhkMTIzMDcxODAxNjBlN2Y2NTRjYjM0NWRlLzMwODc4Y2VhOTg3MDk2NGQ0YTIzODMzOWU5ZGNmZjAwMjA3OGJiYmFhMWEwNThiMDdlMTFjMTY3ZjY3ZWNhMWM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=T7-D9DMNMBzh8F6rP6ONHygAp09VA9venGnWH2Tq%7EOqoSTaRoDCATjBBa5TPdVjHXQQnII3FjSJVqvHWw1TgbMIelMH7OZARrq%7EbEkEJFfZ5r5U9VUAnYeB2Hy52uV96NJWZNH4BD3rGnyJymKiNupvvhQHVpzrozaO7ncs5XZD5N16qqvnUrxFP0iVIG-A84V5G650hLljTge0oA8%7EcQ-fLF7tSYxjwrMvhlY3MNY4fq-2gJDhcHaFaLvFHETrFEkt9dqTzeGWfHFBR5c0KVKXeFnlIfWU2tM4BrKdOPXhStVoqvBQjipSQDIHc9SxSBc06Yiskjoa09L3yDmIJJA__&Key-Pair-Id=KVTP0A1DKRTAX"
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()
        if weight_str == "swizz8_REALBakedvaeFP16.safetensors":
            folder_path = "ComfyUI/models/checkpoints/"
            if os.path.exists(folder_path):
                print(f"The {folder_path} is not exist, create it")
                os.makedirs(folder_path)
            dest = "ComfyUI/models/checkpoints/swizz8_REALBakedvaeFP16.safetensors"
            print("weight_str is swizz8, reset download command")
            subprocess.check_call(
                ["wget", "-O", dest, url], close_fds=False
                # ["pget", "-f", url, dest], close_fds=False
            )
        elif weight_str == "bbox/hand_yolov8s.pt":
            folder_path = "ComfyUI/models/ultralytics/bbox/"
            if os.path.exists(folder_path):
                print(f"The {folder_path} is not exist, create it")
                os.makedirs(folder_path)
            print("weight_str is handYolo, reset download command")
            dest = "ComfyUI/models/ultralytics/bbox/hand_yolov8s.pt"
            subprocess.check_call(
                ["wget", "-O", dest, url], close_fds=False
                # ["pget", "-f", url, dest], close_fds=False
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
