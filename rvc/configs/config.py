import torch
import json
import os

arch_config_paths = {
    "hifi_mrf_refine": [
        os.path.join("hifi_mrf_refine", "48000.json"),
        os.path.join("hifi_mrf_refine", "40000.json"),
        os.path.join("hifi_mrf_refine", "32000.json"),
    ],
    "ringformer": [
        os.path.join("ringformer", "48000.json"),
        os.path.join("ringformer", "24000.json"),
    ],
}

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        initial_precision = self.get_precision()
        
        if self.device == "cpu":
            self.is_half = False
            print("[Config] Running on CPU, forcing fp32 precision.")
        else:
            self.is_half = initial_precision == "bf16"
            print(f"[Config] Running on CUDA, training-only precision loaded from config: {initial_precision}")
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )

        self.json_config = self.load_config_json("hifi_mrf_refine")
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self, vocoder_arch="hifi_mrf_refine"):
        configs = {}
        for config_file in arch_config_paths.get(vocoder_arch, arch_config_paths["hifi_mrf_refine"]):
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs

    def set_precision(self, precision):
        if precision not in ["fp32", "bf16"]:
            raise ValueError("Invalid precision type. Must be 'fp32' or 'bf16'.")

        bf16_run_value = precision == "bf16"
        self.is_half = bf16_run_value

        for config_path in arch_config_paths["hifi_mrf_refine"]:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["bf16_run"] = bf16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")
        return f"Precision set to: {precision}."

    def get_precision(self):
        if not arch_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join("rvc", "configs", arch_config_paths["hifi_mrf_refine"][0])
        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)
            bf16_run_value = config["train"].get("bf16_run", False)
            precision = "bf16" if bf16_run_value else "fp32"
            return precision
        except FileNotFoundError:
            print(f"File not found: {full_config_path}")
            return None

    def check_precision(self):
        if not arch_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join("rvc", "configs", arch_config_paths["hifi_mrf_refine"][0])
        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)
            bf16_run_value = config["train"].get("bf16_run", False)
            precision = "bf16" if bf16_run_value else "fp32"
            runtime_precision = "bf16" if self.is_half else "fp32"
            return {
                "config_precision": precision,
                "runtime_precision": runtime_precision,
                "is_half_flag": self.is_half
            }
        except FileNotFoundError:
            print(f"File not found: {full_config_path}")
            return None


            
    def device_config(self):
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"
            self.is_half = False
            self.set_precision("fp32")

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        low_end_gpus = [
            "16", "P40", "P10", "1050", "1060", "1070", "1080", 
            "2050", "2060", "2070", "2080", "TITAN RTX"
        ]

        if (
            any(gpu_str.lower() in self.gpu_name.lower() for gpu_str in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            if self.is_half:
                print(f"[Config Warning] Your GPU ({self.gpu_name}) does NOT support bf16 precision.")
                print("[Config] Forcing precision to fp32.")
            self.is_half = False
            self.set_precision("fp32")

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024 ** 3)

def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "8"

def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "Unfortunately, there is no compatible GPU available to support your training."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"