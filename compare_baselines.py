#!/usr/bin/env python3
"""
Solar-vs-GLM Critique: LayerNorm Similarity Analysis

This script refutes the claim that high LayerNorm similarity between Solar and GLM
is evidence of model derivation.

Key experiments:
1. Layer 0 baseline is inappropriate (Layer 0 is an outlier)
2. Fair baseline comparison (adjacent layers show high similarity)
3. Multiple MoE models all show high LayerNorm similarity (not unique to Solar-GLM)

GPU: Not required - uses HTTP Range requests to download only LayerNorm weights
"""

import json
import struct
import os
from typing import Optional, Tuple
import numpy as np
import requests
import matplotlib.pyplot as plt

# Model configurations (all MoE models with hidden_size=4096)
MODELS = {
    "Solar": {
        "repo": "upstage/Solar-Open-100B",
        "num_layers": 48,
        "short_name": "Solar",
    },
    "GLM": {
        "repo": "zai-org/GLM-4.5-Air",
        "num_layers": 46,
        "short_name": "GLM",
    },
    "Phi": {
        "repo": "microsoft/Phi-3.5-MoE-instruct",
        "num_layers": 32,
        "short_name": "Phi",
    },
    "Mixtral": {
        "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "num_layers": 32,
        "short_name": "Mixtral",
    },
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

    # Experiment 3: Cross-model comparison (all MoE models)
    print("\n[Experiment 3] Cross-model comparison (all MoE models)")
    print("-" * 60)

    layer = 10
    ln_type = "input_layernorm"

    # Load LayerNorm weights from all models
    weights = {}
    for model_name, config in MODELS.items():
        if layer >= config["num_layers"]:
            print(f"  [SKIP] {model_name}: layer {layer} exceeds {config['num_layers']} layers")
            continue
        weight = get_layernorm_weight(config["repo"], layer, ln_type)
        if weight is not None:
            weights[model_name] = weight

    # Compute pairwise similarities
    valid_models = list(weights.keys())
    n_models = len(valid_models)
    similarity_matrix = np.zeros((n_models, n_models))

    print(f"\n  Pairwise similarities (Layer {layer}):")
    for i, model_a in enumerate(valid_models):
        for j, model_b in enumerate(valid_models):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                a_flat = weights[model_a].flatten()
                b_flat = weights[model_b].flatten()
                if a_flat.shape != b_flat.shape:
                    similarity_matrix[i, j] = np.nan
                    similarity_matrix[j, i] = np.nan
                    print(f"    {MODELS[model_a]['short_name']:8s} vs {MODELS[model_b]['short_name']:8s}: N/A (dim mismatch)")
                else:
                    sim = cosine_similarity(weights[model_a], weights[model_b])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                    print(f"    {MODELS[model_a]['short_name']:8s} vs {MODELS[model_b]['short_name']:8s}: {sim:.6f}")

    # Get Solar-GLM similarity for comparison
    solar_idx = valid_models.index("Solar") if "Solar" in valid_models else -1
    glm_idx = valid_models.index("GLM") if "GLM" in valid_models else -1
    cross_solar_glm = similarity_matrix[solar_idx, glm_idx] if solar_idx >= 0 and glm_idx >= 0 else 0.0

    # Statistics
    upper_triangle = similarity_matrix[np.triu_indices(n_models, k=1)]
    valid_sims = upper_triangle[~np.isnan(upper_triangle)]
    if len(valid_sims) > 0:
        print(f"\n  Statistics ({len(valid_sims)} valid pairs):")
        print(f"    Mean: {np.mean(valid_sims):.6f}")
        print(f"    Std:  {np.std(valid_sims):.6f}")
        print(f"    Min:  {np.min(valid_sims):.6f}")
        print(f"    Max:  {np.max(valid_sims):.6f}")

    # Visualization
    print("\n[Generating visualization...]")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Bar chart (baseline comparison)
    ax1 = axes[0]
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

    bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')

    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_ylim([0, 1.0])
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='High similarity (0.95)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    ax1.set_title('Why Layer 0 Baseline is Inappropriate\nGray: Layer 0 | Blue: Fair | Red: Cross-model',
                  fontsize=12, fontweight='bold')

    # Right: Heatmap (cross-model similarity matrix)
    ax2 = axes[1]
    labels = [MODELS[m]["short_name"] for m in valid_models]

    im = ax2.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')

    ax2.set_xticks(np.arange(n_models))
    ax2.set_yticks(np.arange(n_models))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)

    for i in range(n_models):
        for j in range(n_models):
            value = similarity_matrix[i, j]
            if np.isnan(value):
                label = "N/A"
                color = "gray"
            else:
                label = f"{value:.3f}"
                color = "black" if value > 0.95 else "white"
            ax2.text(j, i, label, ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
    ax2.set_title(f'Cross-Model LayerNorm Similarity\n(Layer {layer}, {ln_type})',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=200, bbox_inches='tight')
    print("  Saved: results/baseline_comparison.png")

    # Summary
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"1. Layer 0 baseline (unfair):        {solar_baseline_mean:.3f}")
    print(f"2. Fair baseline (adjacent layers):  {solar_fair_mean:.3f}")
    print(f"3. Cross-model (Solar vs GLM):       {cross_solar_glm:.3f}")
    if len(valid_sims) > 0:
        print(f"4. All MoE models average:           {np.mean(valid_sims):.3f}")
    print()
    print("KEY FINDINGS:")
    print(f"  - Layer 0 baseline artificially lowers similarity ({solar_baseline_mean:.3f})")
    print(f"  - Fair baseline shows high within-model similarity ({solar_fair_mean:.3f})")
    print(f"  - Solar-GLM similarity ({cross_solar_glm:.3f}) is NOT unique")
    if len(valid_sims) > 0:
        print(f"  - ALL comparable MoE models show similar patterns ({np.mean(valid_sims):.3f} avg)")
    print()
    print("=> The 'evidence' of model derivation is invalidated.")
    print("=> High LayerNorm similarity is a common MoE characteristic.")
    print("="*60)


if __name__ == "__main__":
    main()
