"""Triton chunk scan for Nemotron-H's Mamba-2 layers without a CUDA build.

The Space has no mamba_ssm/causal_conv1d CUDA extensions, so transformers falls back to its PyTorch reference scan. mamba_ssm's
chunk scan (ops/triton/ssd_combined.py) is pure Triton: vendor-mamba holds a `--no-deps` copy installed with
MAMBA_SKIP_CUDA_BUILD. Its package __init__ imports the missing CUDA extension, so the package is registered by path without
running __init__. Only `mamba2_chunk_scan` is swapped; the fused split-conv path (needs causal_conv1d) and the depthwise
convolution stay on the reference code. Import this after transformers' nemotron_h module and before building the model.
"""
import sys, types
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / 'vendor-mamba' / 'mamba_ssm'
ENABLED = False


def enable():
    global ENABLED
    if ENABLED or not VENDOR.is_dir():
        return ENABLED
    import transformers.models.nemotron_h.modeling_nemotron_h as nh
    if 'mamba_ssm' not in sys.modules:
        pkg = types.ModuleType('mamba_ssm'); pkg.__path__ = [str(VENDOR)]; sys.modules['mamba_ssm'] = pkg
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    def chunk_scan(hidden_states, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None,
                   dt_softplus=False, dt_limit=(0.0, float('inf')), return_final_states=False, **_):
        return mamba_chunk_scan_combined(hidden_states, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias,
                                         initial_states=initial_states, dt_softplus=dt_softplus, dt_limit=dt_limit,
                                         return_final_states=return_final_states)
    nh.mamba2_chunk_scan = chunk_scan
    ENABLED = True
    return ENABLED
