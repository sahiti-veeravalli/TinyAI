"""
Configuration and settings for TinyAI backend API.
"""

import os



class Settings:
    PROJECT_NAME: str = "TinyAI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "Quantization, Pruning, and Distillation Pipeline for Enterprise Analytics Edge Devices"
    
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_DIR: str = os.path.join(BASE_DIR, "results")
    MODELS_DIR: str = os.path.join(RESULTS_DIR, "models")
    
    # LLM / AI Research Assistant settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ENABLE_MOCK_LLM_FALLBACK: bool = True  # Allows full offline exploration without external API keys

    # Edge simulator specs (RAM in MB, CPU relative speed, latency budget ms)
    DEVICE_PROFILES = {
        "rpi4": {
            "name": "Raspberry Pi 4B (ARM Cortex-A72)",
            "ram_mb": 4096,
            "max_model_ram_mb": 50,
            "cpu_factor": 2.8,  # ~2.8x slower than reference desktop
            "latency_budget_ms": 15.0,
            "power_budget_w": 5.0
        },
        "jetson_nano": {
            "name": "NVIDIA Jetson Nano (Maxwell GPU + A57)",
            "ram_mb": 2048,
            "max_model_ram_mb": 40,
            "cpu_factor": 2.1,
            "latency_budget_ms": 10.0,
            "power_budget_w": 10.0
        },
        "cortex_m": {
            "name": "ARM Cortex-M55 Microcontroller",
            "ram_mb": 1.0,  # 1024 KB SRAM
            "max_model_ram_mb": 0.35,  # 350 KB
            "cpu_factor": 18.0,
            "latency_budget_ms": 50.0,
            "power_budget_w": 0.25
        },
        "pos_terminal": {
            "name": "Enterprise POS Terminal (Verifone/Ingenico)",
            "ram_mb": 512,
            "max_model_ram_mb": 15,
            "cpu_factor": 4.5,
            "latency_budget_ms": 25.0,
            "power_budget_w": 3.0
        },
        "android_edge": {
            "name": "Low-End Android Handheld (Helio P22)",
            "ram_mb": 1536,
            "max_model_ram_mb": 25,
            "cpu_factor": 3.2,
            "latency_budget_ms": 20.0,
            "power_budget_w": 2.5
        }
    }


settings = Settings()
