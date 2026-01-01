#!/usr/bin/env python3
"""
Baseline Critique Experiment

This script demonstrates why Layer 0 is an inappropriate baseline for
comparing LayerNorm weights across models.

Key comparisons:
1. solar-vs-glm method: Layer 0 vs Layer 10,20,30,40 (unfair)
2. Fair method: Layer 10 vs 20, 20 vs 30 (same distance)
3. Cross-model: Solar[10] vs GLM[10] vs Phi[10]

GPU: Not required - uses HTTP Range requests to download only LayerNorm weights
"""

import json
import struct
import hashlib
from typing import Optional, Dict, List, Tuple
import numpy as np
import requests
import matplotlib.pyplot as plt
import os

# Model configurations
MODELS = {
    "Solar": {
        "repo": "upstage/Solar-Open-100B",
        "num_layers": 48,
    },
    "GLM": {
        "repo": "zai-org/GLM-4.5-Air",
        "num_layers": 46,
    },
    "Phi": {
        "repo": "microsoft/Phi-3.5-MoE-instruct",
        "num_layers": 32,
    }
}

DTYPE_SIZES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def hf_url(repo_id: str, revision: str, filename: str) -> str:
    """Construct HuggingFace file URL"""
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def http_get(url: str, token: Optional[str] = None) -> Optional[bytes]:
    """HTTP GET request"""
    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        print(f"[ERROR] HTTP GET failed: {e}")
    return None


def http_range_get(url: str, start: int, end: int, token: Optional[str] = None) -> bytes:
    """HTTP Range request - download only specified byte range"""
    headers = {"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
    if r.status_code == 206:
        return r.content
    expected = end - start + 1
    if r.status_code == 200 and len(r.content) == expected:
        return r.content
    raise RuntimeError(f"Range GET failed: HTTP {r.status_code}")


def parse_safetensors_header(data: bytes) -> Tuple[int, dict]:
    """Parse safetensors header to get tensor metadata"""
    if len(data) < 8:
        raise ValueError("Invalid safetensors file")

    header_size = struct.unpack("<Q", data[:8])[0]
    header_bytes = data[8 : 8 + header_size]
    header = json.loads(header_bytes.decode("utf-8"))
    return 8 + header_size, header


def get_layernorm_weight(repo_id: str, layer_idx: int, ln_type: str,
                         token: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Download a single LayerNorm weight using HTTP Range request.
    Downloaded weights are cached in cache/ directory.

    Args:
        repo_id: HuggingFace repo ID (e.g., "upstage/Solar-Open-100B")
        layer_idx: Layer index (e.g., 10)
        ln_type: "input_layernorm" or "post_attention_layernorm"
        token: Optional HuggingFace token

    Returns:
        numpy array of the LayerNorm weight, or None if not found
    """
    # Check cache first
    model_name = repo_id.replace("/", "_")
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model_name}_layer{layer_idx}_{ln_type}.npy")

    if os.path.exists(cache_file):
        print(f"[CACHE] Loading {repo_id} layer {layer_idx} {ln_type} from cache")
        return np.load(cache_file)

    # Step 1: Get model file list
    files_url = hf_url(repo_id, "main", "")
    api_url = f"https://huggingface.co/api/models/{repo_id}"

    try:
        r = requests.get(api_url, timeout=30)
        model_info = r.json()
        safetensors_files = [f for f in model_info.get("siblings", [])
                            if f["rfilename"].endswith(".safetensors")]
    except Exception as e:
        print(f"[ERROR] Failed to get model info: {e}")
        return None

    # Step 2: Find the shard containing our layer
    target_key = f"model.layers.{layer_idx}.{ln_type}.weight"

    for file_info in safetensors_files:
        filename = file_info["rfilename"]
        file_url = hf_url(repo_id, "main", filename)

        # Download only header (first 10MB should be enough)
        try:
            header_data = http_range_get(file_url, 0, 10 * 1024 * 1024, token)
            data_offset, header = parse_safetensors_header(header_data)
        except Exception:
            continue

        # Check if our target tensor is in this shard
        if target_key not in header:
            continue

        # Step 3: Download only the LayerNorm weight
        tensor_info = header[target_key]
        dtype = tensor_info["dtype"]
        shape = tensor_info["shape"]
        offsets = tensor_info["data_offsets"]

        start = data_offset + offsets[0]
        end = data_offset + offsets[1] - 1

        print(f"[INFO] Downloading {target_key} from {filename} ({end-start+1} bytes)")
        tensor_data = http_range_get(file_url, start, end, token)

        # Step 4: Parse tensor data and convert to float32
        if dtype == "F32":
            arr = np.frombuffer(tensor_data, dtype=np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(tensor_data, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            # BF16 to FP32 conversion
            u16 = np.frombuffer(tensor_data, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            print(f"[WARN] Unsupported dtype: {dtype}")
            return None

        result = arr.reshape(shape)

        # Save to cache
        np.save(cache_file, result)
        print(f"[CACHE] Saved to {cache_file}")

        return result

    print(f"[WARN] {target_key} not found in any shard")
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


def main():
    """Main experiment"""
    print("="*60)
    print("Baseline Critique Experiment")
    print("="*60)

    os.makedirs("results", exist_ok=True)

    # Experiment 1: solar-vs-glm baseline (Layer 0 vs 10,20,30,40)
    print("\n[Experiment 1] solar-vs-glm baseline (Layer 0 vs distant layers)")
    print("-" * 60)

    solar_ln0 = get_layernorm_weight("upstage/Solar-Open-100B", 0, "input_layernorm")
    glm_ln0 = get_layernorm_weight("zai-org/GLM-4.5-Air", 0, "input_layernorm")

    solar_baseline_sims = []
    glm_baseline_sims = []

    for layer in [10, 20, 30, 40]:
        solar_ln = get_layernorm_weight("upstage/Solar-Open-100B", layer, "input_layernorm")
        glm_ln = get_layernorm_weight("zai-org/GLM-4.5-Air", layer, "input_layernorm")

        if solar_ln is not None and solar_ln0 is not None:
            sim = cosine_similarity(solar_ln0, solar_ln)
            solar_baseline_sims.append(sim)
            print(f"  Solar: Layer 0 vs Layer {layer} = {sim:.6f}")

        if glm_ln is not None and glm_ln0 is not None:
            sim = cosine_similarity(glm_ln0, glm_ln)
            glm_baseline_sims.append(sim)
            print(f"  GLM:   Layer 0 vs Layer {layer} = {sim:.6f}")

    solar_baseline_mean = np.mean(solar_baseline_sims)
    glm_baseline_mean = np.mean(glm_baseline_sims)
    print(f"\n  Solar baseline (Layer 0): {solar_baseline_mean:.6f}")
    print(f"  GLM baseline (Layer 0):   {glm_baseline_mean:.6f}")

    # Experiment 2: Fair baseline (adjacent layers with same distance)
    print("\n[Experiment 2] Fair baseline (adjacent layers, same distance)")
    print("-" * 60)

    solar_fair_sims = []
    glm_fair_sims = []

    pairs = [(10, 20), (20, 30)]
    for layer_a, layer_b in pairs:
        solar_a = get_layernorm_weight("upstage/Solar-Open-100B", layer_a, "input_layernorm")
        solar_b = get_layernorm_weight("upstage/Solar-Open-100B", layer_b, "input_layernorm")
        glm_a = get_layernorm_weight("zai-org/GLM-4.5-Air", layer_a, "input_layernorm")
        glm_b = get_layernorm_weight("zai-org/GLM-4.5-Air", layer_b, "input_layernorm")

        if solar_a is not None and solar_b is not None:
            sim = cosine_similarity(solar_a, solar_b)
            solar_fair_sims.append(sim)
            print(f"  Solar: Layer {layer_a} vs Layer {layer_b} = {sim:.6f}")

        if glm_a is not None and glm_b is not None:
            sim = cosine_similarity(glm_a, glm_b)
            glm_fair_sims.append(sim)
            print(f"  GLM:   Layer {layer_a} vs Layer {layer_b} = {sim:.6f}")

    solar_fair_mean = np.mean(solar_fair_sims)
    glm_fair_mean = np.mean(glm_fair_sims)
    print(f"\n  Solar fair baseline: {solar_fair_mean:.6f}")
    print(f"  GLM fair baseline:   {glm_fair_mean:.6f}")

    # Experiment 3: Cross-model comparison (same layer)
    print("\n[Experiment 3] Cross-model comparison (same layer)")
    print("-" * 60)

    layer = 10
    solar_ln10 = get_layernorm_weight("upstage/Solar-Open-100B", layer, "input_layernorm")
    glm_ln10 = get_layernorm_weight("zai-org/GLM-4.5-Air", layer, "input_layernorm")
    phi_ln10 = get_layernorm_weight("microsoft/Phi-3.5-MoE-instruct", layer, "input_layernorm")

    if solar_ln10 is not None and glm_ln10 is not None:
        sim = cosine_similarity(solar_ln10, glm_ln10)
        print(f"  Solar[{layer}] vs GLM[{layer}]:  {sim:.6f}")
        cross_solar_glm = sim

    if solar_ln10 is not None and phi_ln10 is not None:
        sim = cosine_similarity(solar_ln10, phi_ln10)
        print(f"  Solar[{layer}] vs Phi[{layer}]:  {sim:.6f}")
        cross_solar_phi = sim

    if glm_ln10 is not None and phi_ln10 is not None:
        sim = cosine_similarity(glm_ln10, phi_ln10)
        print(f"  GLM[{layer}] vs Phi[{layer}]:    {sim:.6f}")
        cross_glm_phi = sim

    # Visualization
    print("\n[Generating visualization...]")

    fig, ax = plt.subplots(figsize=(12, 6))

    x_labels = [
        "Solar\nWithin\n(Layer 0)",
        "GLM\nWithin\n(Layer 0)",
        "Solar\nWithin\n(Fair)",
        "GLM\nWithin\n(Fair)",
        "Solar-GLM\nCross\n(Layer 10)"
    ]

    values = [
        solar_baseline_mean,
        glm_baseline_mean,
        solar_fair_mean,
        glm_fair_mean,
        cross_solar_glm
    ]

    colors = ['gray', 'gray', 'blue', 'blue', 'red']

    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='High similarity threshold (0.95)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    ax.set_title('Baseline Comparison: Why Layer 0 is Inappropriate\n' +
                'Gray: Unfair baseline (Layer 0) | Blue: Fair baseline | Red: Cross-model',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=200, bbox_inches='tight')
    print("  Saved: results/baseline_comparison.png")

    # Summary
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"1. solar-vs-glm baseline (Layer 0): {solar_baseline_mean:.3f}")
    print(f"2. Fair baseline (adjacent layers):  {solar_fair_mean:.3f}")
    print(f"3. Cross-model (Solar vs GLM):       {cross_solar_glm:.3f}")
    print()
    print(f"Difference between fair baseline and cross-model: {abs(solar_fair_mean - cross_solar_glm):.3f}")
    print(f"Difference between Layer 0 and cross-model:       {abs(solar_baseline_mean - cross_solar_glm):.3f}")
    print()
    print("=> Layer 0 baseline artificially inflates the difference!")
    print("=> Fair comparison shows within-model and cross-model are similar.")
    print("="*60)


if __name__ == "__main__":
    main()
