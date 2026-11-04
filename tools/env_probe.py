# -*- coding: utf-8 -*-
"""Environment probe for the undistortion TRT feasibility study."""
import sys, platform, json

def main():
    info = {}
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception as e:
        info["numpy"] = f"ERR {e}"
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception as e:
        info["opencv"] = f"ERR {e}"
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_sm"] = f"{p.major}.{p.minor}"
            info["gpu_mem_gib"] = round(p.total_memory / 1024 ** 3, 2)
            info["gpu_sm_count"] = p.multi_processor_count
    except Exception as e:
        info["torch"] = f"ERR {e}"
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except Exception as e:
        info["tensorrt"] = f"ERR {e}"
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
        info["pycuda"] = cuda.get_version()
    except Exception as e:
        info["pycuda"] = f"ERR {e}"
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
