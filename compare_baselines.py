#!/usr/bin/env python3
"""
Solar-vs-GLM Critique: LayerNorm Similarity Analysis

This script refutes the claim that high LayerNorm similarity between Solar and GLM
is evidence of model derivation.

Key experiments:
1. Layer 0 baseline is inappropriate (Layer 0 is an outlier)
2. Fair baseline comparison (adjacent layers show high similarity)
3. Multiple MoE models all show high LayerNorm similarity (not unique to Solar-GLM)
4. Multi-layer consistency (similarity is consistent across all layers, not just specific ones)
5. MoE vs non-MoE comparison (high similarity is MoE-specific, not universal)

GPU: Not required - uses HTTP Range requests to download only LayerNorm weights
"""

import json
import struct
import os
from typing import Optional, Tuple
import numpy as np
import requests
import matplotlib.pyplot as plt

# Model configurations
MOE_MODELS = {
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

# Non-MoE models with hidden_size=4096 for fair comparison
NON_MOE_MODELS_4096 = {
    "Mistral-7B": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "num_layers": 32,
        "hidden_size": 4096,
        "short_name": "Mistral-7B",
    },
    "Yi-1.5-6B": {
        "repo": "01-ai/Yi-1.5-6B-Chat",
        "num_layers": 32,
        "hidden_size": 4096,
        "short_name": "Yi-6B",
    },
    "Yi-1.5-9B": {
        "repo": "01-ai/Yi-1.5-9B-Chat",
        "num_layers": 48,
        "hidden_size": 4096,
        "short_name": "Yi-9B",
    },
    "InternLM2-7B": {
        "repo": "internlm/internlm2-chat-7b",
        "num_layers": 32,
        "hidden_size": 4096,
        "short_name": "InternLM-7B",
    },
    "Qwen1.5-7B": {
        "repo": "Qwen/Qwen1.5-7B-Chat",
        "num_layers": 32,
        "hidden_size": 4096,
        "short_name": "Qwen1.5-7B",
    },
}

# For backward compatibility
NON_MOE_MODELS = NON_MOE_MODELS_4096

# Combined for backward compatibility
MODELS = {**MOE_MODELS, **NON_MOE_MODELS}

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
    print("\n[Experiment 3] Cross-model comparison (MoE models only)")
    print("-" * 60)

    layer = 10
    ln_type = "input_layernorm"

    # Load LayerNorm weights from MoE models only
    weights = {}
    for model_name, config in MOE_MODELS.items():
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
                    print(f"    {MOE_MODELS[model_a]['short_name']:8s} vs {MOE_MODELS[model_b]['short_name']:8s}: N/A (dim mismatch)")
                else:
                    sim = cosine_similarity(weights[model_a], weights[model_b])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                    print(f"    {MOE_MODELS[model_a]['short_name']:8s} vs {MOE_MODELS[model_b]['short_name']:8s}: {sim:.6f}")

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

    # Experiment 4: Multi-layer consistency (MoE models)
    print("\n[Experiment 4] Multi-layer consistency (layers 5,10,15,20,25,30)")
    print("-" * 60)

    layers_to_compare = [5, 10, 15, 20, 25, 30]
    model_pairs = [("Solar", "GLM"), ("Solar", "Phi"), ("Solar", "Mixtral"),
                   ("GLM", "Phi"), ("GLM", "Mixtral"), ("Phi", "Mixtral")]

    # layer -> pair -> similarity
    multi_layer_sims = {layer: {} for layer in layers_to_compare}

    for layer_idx in layers_to_compare:
        print(f"\n  Layer {layer_idx}:")
        layer_weights = {}
        for model_name, config in MOE_MODELS.items():
            if layer_idx < config["num_layers"]:
                w = get_layernorm_weight(config["repo"], layer_idx, "input_layernorm")
                if w is not None:
                    layer_weights[model_name] = w

        for model_a, model_b in model_pairs:
            if model_a in layer_weights and model_b in layer_weights:
                a_flat = layer_weights[model_a].flatten()
                b_flat = layer_weights[model_b].flatten()
                if a_flat.shape == b_flat.shape:
                    sim = cosine_similarity(layer_weights[model_a], layer_weights[model_b])
                    multi_layer_sims[layer_idx][(model_a, model_b)] = sim
                    print(f"    {MOE_MODELS[model_a]['short_name']:8s} vs {MOE_MODELS[model_b]['short_name']:8s}: {sim:.6f}")

    # Compute per-layer averages for plotting
    layer_averages = []
    layer_stds = []
    for layer_idx in layers_to_compare:
        sims = list(multi_layer_sims[layer_idx].values())
        if sims:
            layer_averages.append(np.mean(sims))
            layer_stds.append(np.std(sims))
        else:
            layer_averages.append(np.nan)
            layer_stds.append(np.nan)

    # All multi-layer similarities for statistics
    all_multi_sims = []
    for layer_data in multi_layer_sims.values():
        all_multi_sims.extend(layer_data.values())

    print(f"\n  Multi-layer statistics ({len(all_multi_sims)} comparisons):")
    print(f"    Mean: {np.mean(all_multi_sims):.6f}")
    print(f"    Std:  {np.std(all_multi_sims):.6f}")
    print(f"    Consistency (std of layer means): {np.std(layer_averages):.6f}")

    # Experiment 5: Hidden size comparison (same hidden_size=4096)
    print("\n[Experiment 5] Same hidden_size (4096) comparison (Layer 10)")
    print("-" * 60)

    # Load non-MoE model weights (all hidden_size=4096)
    non_moe_weights = {}
    print("\n  Loading non-MoE models (hidden_size=4096):")
    for model_name, config in NON_MOE_MODELS.items():
        if layer < config["num_layers"]:
            w = get_layernorm_weight(config["repo"], layer, ln_type)
            if w is not None:
                non_moe_weights[model_name] = w
                print(f"    {config['short_name']:12s}: shape={w.shape}")

    # MoE vs MoE (baseline)
    moe_vs_moe_sims = []
    print("\n  MoE vs MoE (all hidden_size=4096):")
    for i, model_a in enumerate(valid_models):
        for j, model_b in enumerate(valid_models):
            if i < j:
                sim = similarity_matrix[i, j]
                if not np.isnan(sim):
                    moe_vs_moe_sims.append(sim)
                    print(f"    {MOE_MODELS[model_a]['short_name']:8s} vs {MOE_MODELS[model_b]['short_name']:8s}: {sim:.6f}")

    # MoE vs non-MoE (all combinations)
    moe_vs_nonmoe_sims = []
    moe_vs_nonmoe_details = []
    print("\n  MoE vs non-MoE (all hidden_size=4096):")
    for moe_name in valid_models:
        moe_w = weights[moe_name]
        for nonmoe_name, nonmoe_w in non_moe_weights.items():
            if moe_w.flatten().shape == nonmoe_w.flatten().shape:
                sim = cosine_similarity(moe_w, nonmoe_w)
                moe_vs_nonmoe_sims.append(sim)
                moe_vs_nonmoe_details.append((moe_name, nonmoe_name, sim))
                print(f"    {MOE_MODELS[moe_name]['short_name']:8s} vs {NON_MOE_MODELS[nonmoe_name]['short_name']:12s}: {sim:.6f}")

    # non-MoE vs non-MoE
    nonmoe_vs_nonmoe_sims = []
    nonmoe_models = list(non_moe_weights.keys())
    print("\n  non-MoE vs non-MoE (all hidden_size=4096):")
    for i, model_a in enumerate(nonmoe_models):
        for j, model_b in enumerate(nonmoe_models):
            if i < j:
                w_a = non_moe_weights[model_a]
                w_b = non_moe_weights[model_b]
                if w_a.flatten().shape == w_b.flatten().shape:
                    sim = cosine_similarity(w_a, w_b)
                    nonmoe_vs_nonmoe_sims.append(sim)
                    print(f"    {NON_MOE_MODELS[model_a]['short_name']:12s} vs {NON_MOE_MODELS[model_b]['short_name']:12s}: {sim:.6f}")

    # Statistics
    print(f"\n  Summary:")
    print(f"    MoE vs MoE:         {np.mean(moe_vs_moe_sims):.4f} ± {np.std(moe_vs_moe_sims):.4f} (n={len(moe_vs_moe_sims)})")
    if moe_vs_nonmoe_sims:
        print(f"    MoE vs non-MoE:     {np.mean(moe_vs_nonmoe_sims):.4f} ± {np.std(moe_vs_nonmoe_sims):.4f} (n={len(moe_vs_nonmoe_sims)})")
    if nonmoe_vs_nonmoe_sims:
        print(f"    non-MoE vs non-MoE: {np.mean(nonmoe_vs_nonmoe_sims):.4f} ± {np.std(nonmoe_vs_nonmoe_sims):.4f} (n={len(nonmoe_vs_nonmoe_sims)})")

    # Visualization - Individual images for each experiment
    print("\n[Generating visualizations...]")

    # ============================================================
    # Experiment 1: Layer 0 Outlier Visualization
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Show Layer 0 vs other layers for both models
    layer_indices = [0, 10, 20, 30, 40]
    solar_layer0_sims = [1.0] + solar_baseline_sims  # Layer 0 vs itself = 1.0
    glm_layer0_sims = [1.0] + glm_baseline_sims

    x = np.arange(len(layer_indices))
    width = 0.35

    bars1 = ax1.bar(x - width/2, solar_layer0_sims, width, label='Solar', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, glm_layer0_sims, width, label='GLM', color='#3498db', alpha=0.8)

    ax1.set_xlabel('Target Layer', fontsize=12)
    ax1.set_ylabel('Cosine Similarity with Layer 0', fontsize=12)
    ax1.set_title('Layer 0 is an Outlier\nSimilarity drops dramatically from Layer 0 to other layers', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Layer {i}' for i in layer_indices])
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='0.95 threshold')
    ax1.axhline(y=np.mean(solar_baseline_sims), color='gray', linestyle=':', alpha=0.7, label=f'Mean (~{np.mean(solar_baseline_sims):.2f})')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp1_layer0_outlier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp1_layer0_outlier.png")

    # ============================================================
    # Experiment 2: Fair Baseline (Adjacent Layers)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    categories = ['Layer 10↔20', 'Layer 20↔30', 'Mean']
    solar_vals = solar_fair_sims + [solar_fair_mean]
    glm_vals = glm_fair_sims + [glm_fair_mean]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, solar_vals, width, label='Solar', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x + width/2, glm_vals, width, label='GLM', color='#3498db', alpha=0.8)

    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Fair Baseline: Adjacent Layer Similarity\nHigh similarity (~0.99) between neighboring layers', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim([0.95, 1.005])
    ax2.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='0.99')
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp2_fair_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp2_fair_baseline.png")

    # ============================================================
    # Experiment 3: MoE Model Heatmap
    # ============================================================
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    labels = [MOE_MODELS[m]["short_name"] for m in valid_models]

    im = ax3.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')

    ax3.set_xticks(np.arange(n_models))
    ax3.set_yticks(np.arange(n_models))
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.set_yticklabels(labels, fontsize=11)

    for i in range(n_models):
        for j in range(n_models):
            value = similarity_matrix[i, j]
            if np.isnan(value):
                lbl = "N/A"
                color = "gray"
            else:
                lbl = f"{value:.3f}"
                color = "black" if value > 0.95 else "white"
            ax3.text(j, i, lbl, ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
    ax3.set_title(f'MoE Models LayerNorm Similarity (Layer {layer})\nAll pairs show high similarity (>0.94)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp3_moe_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp3_moe_heatmap.png")

    # ============================================================
    # Experiment 4: Multi-Layer Consistency
    # ============================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.errorbar(layers_to_compare, layer_averages, yerr=layer_stds,
                 marker='o', linewidth=2, markersize=10, capsize=5,
                 color='#3498db', label='Mean ± Std', capthick=2)
    ax4.fill_between(layers_to_compare,
                     np.array(layer_averages) - np.array(layer_stds),
                     np.array(layer_averages) + np.array(layer_stds),
                     alpha=0.2, color='#3498db')
    ax4.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='0.95 threshold', alpha=0.7)
    ax4.set_xlabel('Layer Index', fontsize=12)
    ax4.set_ylabel('Cosine Similarity', fontsize=12)
    ax4.set_ylim([0.9, 1.0])
    ax4.set_xlim([0, 35])
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.set_title('Multi-Layer Consistency\nLayerNorm similarity is stable across all layers', fontsize=13, fontweight='bold')

    # Add value labels
    for i, (x_val, y_val) in enumerate(zip(layers_to_compare, layer_averages)):
        ax4.annotate(f'{y_val:.3f}', (x_val, y_val), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp4_multi_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp4_multi_layer.png")

    # ============================================================
    # Experiment 5: Architecture Comparison (Bar Chart)
    # ============================================================
    fig5, ax5 = plt.subplots(figsize=(9, 6))

    categories = ['MoE vs MoE', 'MoE vs non-MoE', 'non-MoE vs non-MoE']
    means = [
        np.mean(moe_vs_moe_sims) if moe_vs_moe_sims else 0,
        np.mean(moe_vs_nonmoe_sims) if moe_vs_nonmoe_sims else 0,
        np.mean(nonmoe_vs_nonmoe_sims) if nonmoe_vs_nonmoe_sims else 0
    ]
    stds = [
        np.std(moe_vs_moe_sims) if moe_vs_moe_sims else 0,
        np.std(moe_vs_nonmoe_sims) if moe_vs_nonmoe_sims else 0,
        np.std(nonmoe_vs_nonmoe_sims) if nonmoe_vs_nonmoe_sims else 0
    ]
    counts = [len(moe_vs_moe_sims), len(moe_vs_nonmoe_sims), len(nonmoe_vs_nonmoe_sims)]
    colors_exp5 = ['#2ecc71', '#f39c12', '#9b59b6']

    bars5 = ax5.bar(range(len(categories)), means, yerr=stds, capsize=8,
                    color=colors_exp5, alpha=0.8, edgecolor='black', linewidth=1.5)

    for i, (bar, val) in enumerate(zip(bars5, means)):
        if val > 0:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                    f'{val:.4f}\n(n={counts[i]})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax5.set_xticks(range(len(categories)))
    ax5.set_xticklabels(categories, fontsize=12)
    ax5.set_ylabel('Cosine Similarity', fontsize=12)
    ax5.set_ylim([0.9, 1.05])
    ax5.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=2, label='0.95 threshold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.legend(loc='lower right')
    ax5.set_title('Architecture Comparison (All hidden_size=4096)\nNo significant difference between MoE and non-MoE', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp5_architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp5_architecture_comparison.png")

    # ============================================================
    # Experiment 5: Full 9-Model Heatmap
    # ============================================================
    # Build full similarity matrix for all 9 models
    all_model_names = list(valid_models) + list(non_moe_weights.keys())
    all_weights = {**{m: weights[m] for m in valid_models}, **non_moe_weights}
    n_all = len(all_model_names)
    full_similarity_matrix = np.zeros((n_all, n_all))

    for i, model_a in enumerate(all_model_names):
        for j, model_b in enumerate(all_model_names):
            if i == j:
                full_similarity_matrix[i, j] = 1.0
            else:
                w_a = all_weights[model_a].flatten()
                w_b = all_weights[model_b].flatten()
                if w_a.shape == w_b.shape:
                    full_similarity_matrix[i, j] = cosine_similarity(all_weights[model_a], all_weights[model_b])
                else:
                    full_similarity_matrix[i, j] = np.nan

    fig6, ax6 = plt.subplots(figsize=(12, 10))

    # Get short names
    all_labels = []
    for m in all_model_names:
        if m in MOE_MODELS:
            all_labels.append(MOE_MODELS[m]["short_name"] + " (MoE)")
        else:
            all_labels.append(NON_MOE_MODELS[m]["short_name"])

    im = ax6.imshow(full_similarity_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')

    ax6.set_xticks(np.arange(n_all))
    ax6.set_yticks(np.arange(n_all))
    ax6.set_xticklabels(all_labels, fontsize=9, rotation=45, ha='right')
    ax6.set_yticklabels(all_labels, fontsize=9)

    for i in range(n_all):
        for j in range(n_all):
            value = full_similarity_matrix[i, j]
            if np.isnan(value):
                lbl = "N/A"
                color = "gray"
            else:
                lbl = f"{value:.2f}"
                color = "black" if value > 0.95 else "white"
            ax6.text(j, i, lbl, ha="center", va="center", color=color, fontsize=8, fontweight='bold')

    # Draw separator lines between MoE and non-MoE
    n_moe = len(valid_models)
    ax6.axhline(y=n_moe - 0.5, color='white', linewidth=3)
    ax6.axvline(x=n_moe - 0.5, color='white', linewidth=3)

    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
    ax6.set_title(f'Full Model Similarity Matrix (Layer {layer}, hidden_size=4096)\nSolar-GLM similarity is NOT unique - all models show similar patterns',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp5_full_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/exp5_full_heatmap.png")

    # ============================================================
    # Summary: Combined 4-panel figure (for overview)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Layer 0 outlier
    ax = axes[0, 0]
    x = np.arange(len(layer_indices))
    width = 0.35
    ax.bar(x - width/2, solar_layer0_sims, width, label='Solar', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, glm_layer0_sims, width, label='GLM', color='#3498db', alpha=0.8)
    ax.set_xlabel('Target Layer')
    ax.set_ylabel('Cosine Similarity with Layer 0')
    ax.set_title('Exp 1: Layer 0 is an Outlier', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layer_indices])
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: MoE heatmap
    ax = axes[0, 1]
    labels = [MOE_MODELS[m]["short_name"] for m in valid_models]
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(n_models):
        for j in range(n_models):
            value = similarity_matrix[i, j]
            lbl = f"{value:.2f}" if not np.isnan(value) else "N/A"
            color = "black" if value > 0.95 else "white"
            ax.text(j, i, lbl, ha="center", va="center", color=color, fontsize=9, fontweight='bold')
    ax.set_title('Exp 3: MoE Models Similarity', fontweight='bold')

    # Panel 3: Multi-layer
    ax = axes[1, 0]
    ax.errorbar(layers_to_compare, layer_averages, yerr=layer_stds,
                marker='o', linewidth=2, markersize=8, capsize=5, color='#3498db')
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, alpha=0.3)
    ax.set_title('Exp 4: Multi-Layer Consistency', fontweight='bold')

    # Panel 4: Architecture comparison
    ax = axes[1, 1]
    bars = ax.bar(range(len(categories)), means, yerr=stds, capsize=5,
                  color=colors_exp5, alpha=0.8, edgecolor='black')
    for i, (bar, val) in enumerate(zip(bars, means)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + stds[i] + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(['MoE-MoE', 'MoE-nonMoE', 'nonMoE-nonMoE'], fontsize=9)
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim([0.9, 1.05])
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Exp 5: Architecture Comparison', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/summary_comparison.png")

    # Summary
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"1. Layer 0 baseline (unfair):        {solar_baseline_mean:.3f}")
    print(f"2. Fair baseline (adjacent layers):  {solar_fair_mean:.3f}")
    print(f"3. Cross-model (Solar vs GLM):       {cross_solar_glm:.3f}")
    if len(valid_sims) > 0:
        print(f"4. All MoE models average:           {np.mean(valid_sims):.3f}")
    if all_multi_sims:
        print(f"5. Multi-layer consistency:          {np.mean(all_multi_sims):.3f} (std: {np.std(layer_averages):.4f})")
    print(f"6. MoE vs MoE (4096):                {np.mean(moe_vs_moe_sims):.3f} (n={len(moe_vs_moe_sims)})")
    if moe_vs_nonmoe_sims:
        print(f"7. MoE vs non-MoE (4096):            {np.mean(moe_vs_nonmoe_sims):.3f} (n={len(moe_vs_nonmoe_sims)})")
    if nonmoe_vs_nonmoe_sims:
        print(f"8. non-MoE vs non-MoE (4096):        {np.mean(nonmoe_vs_nonmoe_sims):.3f} (n={len(nonmoe_vs_nonmoe_sims)})")
    print()
    print("OBSERVATIONS:")
    print(f"  - Layer 0 기준 유사도: {solar_baseline_mean:.3f}")
    print(f"  - 인접 레이어 유사도: {solar_fair_mean:.3f}")
    print(f"  - Solar-GLM 유사도: {cross_solar_glm:.3f}")
    if len(valid_sims) > 0:
        print(f"  - MoE 모델 간 평균: {np.mean(valid_sims):.3f}")
    if all_multi_sims:
        print(f"  - 레이어별 표준편차: {np.std(layer_averages):.4f}")
    if moe_vs_nonmoe_sims:
        print(f"  - MoE vs non-MoE (4096): {np.mean(moe_vs_nonmoe_sims):.3f}")
    if nonmoe_vs_nonmoe_sims:
        print(f"  - non-MoE vs non-MoE (4096): {np.mean(nonmoe_vs_nonmoe_sims):.3f}")
    print("="*60)

    # Save results as Markdown
    from datetime import datetime

    md_content = f"""# LayerNorm 유사도 비교 실험 결과

> 생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 개요

Solar와 GLM 모델 간 LayerNorm 가중치 유사도를 다양한 조건에서 측정한 결과입니다.

![실험 결과](baseline_comparison.png)

---

## 실험 1: Layer 0 기준 유사도

Layer 0을 기준으로 다른 레이어들과의 유사도를 측정한 결과입니다.

| 모델 | Layer 0 vs 10 | Layer 0 vs 20 | Layer 0 vs 30 | Layer 0 vs 40 | 평균 |
|------|---------------|---------------|---------------|---------------|------|
| Solar | {solar_baseline_sims[0]:.4f} | {solar_baseline_sims[1]:.4f} | {solar_baseline_sims[2]:.4f} | {solar_baseline_sims[3]:.4f} | {solar_baseline_mean:.4f} |
| GLM   | {glm_baseline_sims[0]:.4f} | {glm_baseline_sims[1]:.4f} | {glm_baseline_sims[2]:.4f} | {glm_baseline_sims[3]:.4f} | {glm_baseline_mean:.4f} |

---

## 실험 2: 인접 레이어 간 유사도

동일한 거리(10 레이어)를 가진 인접 레이어 쌍의 유사도입니다.

| 모델 | Layer 10 vs 20 | Layer 20 vs 30 | 평균 |
|------|----------------|----------------|------|
| Solar | {solar_fair_sims[0]:.4f} | {solar_fair_sims[1]:.4f} | {solar_fair_mean:.4f} |
| GLM   | {glm_fair_sims[0]:.4f} | {glm_fair_sims[1]:.4f} | {glm_fair_mean:.4f} |

---

## 실험 3: MoE 모델 간 비교 (Layer {layer})

4개의 MoE 모델에서 동일 레이어의 LayerNorm 유사도입니다.

| 모델 쌍 | Cosine 유사도 |
|---------|---------------|
| Solar vs GLM | {similarity_matrix[valid_models.index("Solar"), valid_models.index("GLM")]:.4f} |
| Solar vs Phi | {similarity_matrix[valid_models.index("Solar"), valid_models.index("Phi")]:.4f} |
| Solar vs Mixtral | {similarity_matrix[valid_models.index("Solar"), valid_models.index("Mixtral")]:.4f} |
| GLM vs Phi | {similarity_matrix[valid_models.index("GLM"), valid_models.index("Phi")]:.4f} |
| GLM vs Mixtral | {similarity_matrix[valid_models.index("GLM"), valid_models.index("Mixtral")]:.4f} |
| Phi vs Mixtral | {similarity_matrix[valid_models.index("Phi"), valid_models.index("Mixtral")]:.4f} |

- 평균: {np.mean(valid_sims):.4f}
- 표준편차: {np.std(valid_sims):.4f}

---

## 실험 4: 레이어별 유사도 변화

여러 레이어(5, 10, 15, 20, 25, 30)에서 MoE 모델 간 유사도 추이입니다.

| Layer | 평균 유사도 | 표준편차 |
|-------|-------------|----------|
"""
    for i, layer_idx in enumerate(layers_to_compare):
        md_content += f"| {layer_idx} | {layer_averages[i]:.4f} | {layer_stds[i]:.4f} |\n"

    md_content += f"""
- 전체 비교 횟수: {len(all_multi_sims)}회
- 전체 평균: {np.mean(all_multi_sims):.4f}
- 전체 표준편차: {np.std(all_multi_sims):.4f}
- 레이어별 평균의 표준편차: {np.std(layer_averages):.4f}

---

## 실험 5: 동일 hidden_size (4096) 비교

동일 hidden_size(4096)를 가진 MoE와 non-MoE 모델 간 유사도 비교입니다.

### 비교 결과

| 비교 유형 | 평균 유사도 | 표준편차 | 비교 횟수 |
|-----------|-------------|----------|-----------|
| MoE vs MoE | {np.mean(moe_vs_moe_sims):.4f} | {np.std(moe_vs_moe_sims):.4f} | {len(moe_vs_moe_sims)} |
| MoE vs non-MoE | {f'{np.mean(moe_vs_nonmoe_sims):.4f}' if moe_vs_nonmoe_sims else 'N/A'} | {f'{np.std(moe_vs_nonmoe_sims):.4f}' if moe_vs_nonmoe_sims else 'N/A'} | {len(moe_vs_nonmoe_sims)} |
| non-MoE vs non-MoE | {f'{np.mean(nonmoe_vs_nonmoe_sims):.4f}' if nonmoe_vs_nonmoe_sims else 'N/A'} | {f'{np.std(nonmoe_vs_nonmoe_sims):.4f}' if nonmoe_vs_nonmoe_sims else 'N/A'} | {len(nonmoe_vs_nonmoe_sims)} |

**참고**: 모든 모델이 hidden_size=4096으로 동일하여 Cosine similarity 계산이 가능합니다.

---

## 결과 요약

| 측정 항목 | 값 |
|-----------|-----|
| Layer 0 기준 유사도 (Solar) | {solar_baseline_mean:.4f} |
| Layer 0 기준 유사도 (GLM) | {glm_baseline_mean:.4f} |
| 인접 레이어 유사도 (Solar) | {solar_fair_mean:.4f} |
| 인접 레이어 유사도 (GLM) | {glm_fair_mean:.4f} |
| Solar vs GLM (Layer 10) | {cross_solar_glm:.4f} |
| MoE 모델 간 평균 | {np.mean(valid_sims):.4f} |
| MoE vs non-MoE 평균 | {f'{np.mean(moe_vs_nonmoe_sims):.4f}' if moe_vs_nonmoe_sims else 'N/A'} |
| non-MoE vs non-MoE 평균 | {f'{np.mean(nonmoe_vs_nonmoe_sims):.4f}' if nonmoe_vs_nonmoe_sims else 'N/A'} |
| 레이어별 유사도 표준편차 | {np.std(layer_averages):.4f} |

---

## 사용된 모델

### MoE 모델 (hidden_size=4096)
- Solar-Open-100B (Upstage)
- GLM-4.5-Air (Zhipu AI)
- Phi-3.5-MoE-instruct (Microsoft)
- Mixtral-8x7B-Instruct-v0.1 (Mistral AI)

### non-MoE 모델 (hidden_size=4096)
- Mistral-7B-Instruct-v0.3 (Mistral AI)
- Yi-1.5-6B-Chat (01.AI)
- Yi-1.5-9B-Chat (01.AI)
- InternLM2-chat-7B (Shanghai AI Lab)
- Qwen1.5-7B-Chat (Alibaba)

---

## 실험 환경

- 측정 대상: `input_layernorm.weight`
- 유사도 측정: Cosine Similarity
- 가중치 추출: HTTP Range Request (전체 모델 다운로드 없이 LayerNorm만 추출)
"""

    with open("results/RESULTS.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("\n  Saved: results/RESULTS.md")


if __name__ == "__main__":
    main()
