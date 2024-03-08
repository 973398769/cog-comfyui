import json
import os
import shutil
import tarfile
import zipfile
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("examples/photomaker.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path):
        file_extension = os.path.splitext(input_file)[1]
        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.log_and_collect_files(INPUT_DIR)
        print("====================================")

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        function_name: str = Input(
            description="The specific function you need, such as: hand_restoration, face_restoration",
            choices=['hand_restoration', 'face_restoration', 'all'],
            default="hand_restoration",
        ),
        workflow_json: str = Input(
            description="Your ComfyUI workflow as JSON. You must use the API version of your workflow. Get it from ComfyUI using ‘Save (API format)’. Instructions here: https://github.com/fofr/cog-comfyui",
            default="",
        ),
        input_file: Path = Input(
            description="Input image, tar or zip file. Read guidance on workflows and input files here: https://github.com/fofr/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them.",
            default=None,
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
        randomise_seeds: bool = Input(
            description="Automatically randomise seeds (seed, noise_seed, rand_seed)",
            default=True,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if input_file:
            self.handle_input_file(input_file)

        # TODO: Record the previous models loaded
        # If different, run /free to free up models and memory

        if not workflow_json:
            print("import workflow from examples path")
            workflow_json = choose_workflow(function_name, input_file)
        check_custom_nodes()
        wf = self.comfyUI.load_workflow(workflow_json or EXAMPLE_WORKFLOW_JSON)

        if randomise_seeds:
            self.comfyUI.randomise_seeds(wf)

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        return files

def check_custom_nodes():
    items = os.listdir("ComfyUI/custom_nodes/")
    for item in items:
        print(f"custom node is {item}")
        if item == "facerestore_cf":
            print("facerestore_cf is exist")
            facerestore_items = os.listdir("ComfyUI/custom_nodes/facerestore_cf/")
            for facerestore_item in facerestore_items:
                print(f"facerestore_item is {facerestore_item}")
                if facerestore_item == "__init__.py":
                    with open("ComfyUI/custom_nodes/facerestore_cf/__init__.py") as it:
                        print(it.read())

def choose_workflow(function_name, input_file):
    workflow_json = json.dumps({})
    if function_name == "hand_restoration":
        with open("examples/hands_restoration_api.json", "r") as file:
            workflow_json = json.load(file)
            workflow_json["57"]["inputs"]["image"] = input_file
    if function_name == "face_restoration":
        with open("examples/faces_restoration_api.json", "r") as file:
            workflow_json = json.load(file)
            workflow_json["3"]["inputs"]["image"] = input_file
    return workflow_json