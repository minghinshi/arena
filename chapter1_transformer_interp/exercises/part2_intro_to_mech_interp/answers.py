# %% Setup
import functools
import math
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    TransformerBridgeConfig,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from part2_intro_to_mech_interp.shortformer import ShortformerConfig, ShortformerTransformer
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# --- Section 1 ---

# %% Load GPT-2
gpt2_small: TransformerBridge = TransformerBridge.boot_transformers("gpt2")

# %% Exercise 1.1: Inspect
gpt2_small.cfg

# Layers:           12
# Heads:            12
# Context window: 1024

# %% Run the model
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)

# %% Exercise 1.2: Calculate accuracy
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
labels = gpt2_small.to_tokens(model_description_text)[0, 1:]
print(f"Correct tokens: {t.sum(prediction == labels).item()}")

# %% Cache activations
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

# %% Exercise 1.3: Verify activations
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

layer0_q = gpt2_cache["q", 0]
layer0_k = gpt2_cache["k", 0]

# Multiply
attn_scores = einops.einsum(
    layer0_q,
    layer0_k,
    "q_pos head dim, k_pos head dim -> head q_pos k_pos",
)

# Scale
attn_scores = attn_scores / math.sqrt(gpt2_small.cfg.d_head)

# Mask
seq_len = layer0_q.size(0)
causal_mask = t.ones((seq_len, seq_len), device=layer0_q.device).triu(diagonal=1).bool()
attn_scores.masked_fill_(causal_mask, float("-inf"))

# Softmax
layer0_pattern_from_q_and_k = F.softmax(attn_scores, dim=-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %% See attention patterns
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

# --- Section 2 ---

# %% Define toy transformer
cfg = ShortformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
)

# %% Download the weights
from huggingface_hub import hf_hub_download

REPO_ID = "arena-education/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# %% Create model and load weights
model = ShortformerTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %% Exercise 2.1: Visualise and inspect attention patterns
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)

attention_pattern = cache["pattern", 0]
display(
    cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

attention_pattern = cache["pattern", 1]
display(
    cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L1H{i}" for i in range(12)],
    )
)


# %% Exercise 2.2: Head detectors
def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    heads: list[str] = []
    for layer_idx in range(cfg.n_layers):
        for head_idx in range(cfg.n_heads):
            if t.diagonal(cache["pattern", layer_idx][head_idx]).mean() >= 0.5:
                heads.append(f"{layer_idx}.{head_idx}")
    return heads


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    heads: list[str] = []
    for layer_idx in range(cfg.n_layers):
        for head_idx in range(cfg.n_heads):
            if t.diagonal(cache["pattern", layer_idx][head_idx], offset=-1).mean() >= 0.5:
                heads.append(f"{layer_idx}.{head_idx}")
    return heads


def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    heads: list[str] = []
    for layer_idx in range(cfg.n_layers):
        for head_idx in range(cfg.n_heads):
            if t.mean(cache["pattern", layer_idx][head_idx, :, 0]) >= 0.5:
                heads.append(f"{layer_idx}.{head_idx}")
    return heads


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%
