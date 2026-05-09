"""
deployment_tiers.py — TEN-GNN Optimization Roadmap
=================================================

This file documents the technical requirements and expected performance 
characteristics for each deployment tier of the spoofing detection system.
Targets are based on the HFT market surveillance benchmarks.
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TierConfig:
    name: str
    precision: str
    hardware: str
    target_latency: str
    optimization_method: str
    suitable_for: str

DEPLOYMENT_TIERS = {
    "Tier 1: Baseline": TierConfig(
        name="Baseline",
        precision="FP32",
        hardware="CPU (x86)",
        target_latency="~30ms",
        optimization_method="Python Eager Mode",
        suitable_for="Local research, backtesting (slow)"
    ),
    "Tier 2: Production-Lite": TierConfig(
        name="TorchScript",
        precision="FP32",
        hardware="CPU (x86)",
        target_latency="~15-20ms",
        optimization_method="TorchScript Graph Optimization",
        suitable_for="Near-real-time monitoring, high-throughput cloud"
    ),
    "Tier 3: Optimized CPU": TierConfig(
        name="Quantized CPU",
        precision="INT8",
        hardware="CPU (AVX-512)",
        target_latency="~10ms",
        optimization_method="Dynamic Quantization (PyTorch)",
        suitable_for="Standard market surveillance (100ms budget)"
    ),
    "Tier 4: GPU Accelerator": TierConfig(
        name="TensorRT",
        precision="INT8 / FP16",
        hardware="NVIDIA GPU (CUDA)",
        target_latency="1-2ms",
        optimization_method="TensorRT Engine + Mixed Precision",
        suitable_for="Low-latency exchange surveillance"
    ),
    "Tier 5: Ultra-Low Latency": TierConfig(
        name="FPGA Hybrid",
        precision="INT8",
        hardware="FPGA + GPU (HBM)",
        target_latency="0.6ms (600µs)",
        optimization_method="Logic Gates (VHDL) + TensorRT",
        suitable_for="HFT Tick-to-Trade Protection (Sarao-class)"
    )
}

def print_roadmap():
    print("="*80)
    print(f"{'Tier':<25} | {'Latency':<10} | {'Method':<25}")
    print("-" * 80)
    for tier in DEPLOYMENT_TIERS.values():
        print(f"{tier.name:<25} | {tier.target_latency:<10} | {tier.optimization_method:<25}")
    print("=" * 80)

if __name__ == "__main__":
    print_roadmap()
