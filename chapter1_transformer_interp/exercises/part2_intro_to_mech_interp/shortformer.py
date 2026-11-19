"""
A two-layer attention-only transformer with shortformer positional embeddings.

Written by GLM 5.3 Flash. Edited by minghinshi.

TransformerLens v4 removed the `HookedTransformer` / `HookedTransformerConfig` classes
and no longer implements `positional_embedding_type="shortformer"` (the v4 native model
only supports `"standard"` and `"rotary"`, and `boot_tl_legacy` explicitly refuses
shortformer checkpoints). This module is a self-contained replacement for the toy model
used in section 2 of the ARENA material ("Finding induction heads"), which was trained
with shortformer-style positional embeddings.

Shortformer semantics (as in TransformerLens <= 3.x): the positional embedding is NOT
added to the residual stream. Instead it is added to the input of the query and key
computations only, i.e. `Q = (resid + pos_embed) @ W_Q + b_Q` and same for keys, but
`V = resid @ W_V + b_V`. This means the residual stream can't directly encode positional
information, which makes induction heads form much more easily.

The model subclasses `HookedRootModule`, which TransformerLens v4 explicitly keeps as
the supported way to hook arbitrary `nn.Module`s. Hook names follow the TransformerLens
conventions, so everything the exercises rely on works out of the box:

- `model.run_with_cache(...) -> (logits, ActivationCache)` with shorthand indexing,
  e.g. `cache["pattern", 1]`, `cache["q", 0]`, `cache["embed"]`, `cache["pos_embed"]`,
  `cache["result", 0]`, `cache["resid_pre", 1]`, `cache["z", 0]`, plus
  `cache.remove_batch_dim()`.
- `model.run_with_hooks(tokens, return_type=None, fwd_hooks=[...])`, with hooks named
  via `transformer_lens.utilities.get_act_name` (e.g. `get_act_name("z", 0)` ->
  `"blocks.0.attn.hook_z"`) and filters like `lambda name: name.endswith("pattern")`.
- `hook.layer()` on `HookPoint`s, and `hook.ctx` for stashing results.
- Weight accessors with TransformerLens shapes: `model.W_Q[layer, head]` ->
  `[d_model, d_head]`, `model.W_O[layer, head]` -> `[d_head, d_model]`, plus
  `model.W_E`, `model.W_U`, `model.W_pos`, `model.cfg`.

The state dict of the pretrained checkpoint (keys `embed.W_E`, `pos_embed.W_pos`,
`blocks.{i}.attn.*`, `unembed.*`) loads directly with `model.load_state_dict(...)`.

Usage (drop-in replacement for the material's `HookedTransformer` section):

    cfg = ShortformerConfig(
        d_model=768, d_head=64, n_heads=12, n_layers=2, n_ctx=2048, d_vocab=50278,
        tokenizer_name="EleutherAI/gpt-neox-20b", seed=398, use_attn_result=True,
    )
    model = ShortformerTransformer(cfg)
    pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(pretrained_weights)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import einops
import torch as t
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformer_lens import ActivationCache, HookedRootModule
from transformer_lens.hook_points import HookPoint


@dataclass
class ShortformerConfig:
    """Minimal config, mirroring the `HookedTransformerConfig` args the material uses."""

    d_model: int
    d_head: int
    n_heads: int
    n_layers: int
    n_ctx: int
    d_vocab: int

    tokenizer_name: str | None = None
    attention_dir: Literal["causal", "bidirectional"] = "causal"
    attn_only: bool = True
    use_attn_result: bool = True
    # Only "shortformer" is implemented; kept as a field so the material's config
    # arguments can be passed through unchanged.
    positional_embedding_type: str = "shortformer"
    default_prepend_bos: bool = True
    seed: int | None = None
    # If None, picks cuda > mps > cpu. Stored as a string, like HookedTransformerConfig.
    device: str | None = None

    def __post_init__(self):
        assert self.positional_embedding_type == "shortformer", (
            f"Only positional_embedding_type='shortformer' is implemented "
            f"(got {self.positional_embedding_type!r})."
        )
        assert self.attention_dir == "causal", "Only causal attention is implemented."
        if self.device is None:
            self.device = (
                "cuda"
                if t.cuda.is_available()
                else "mps"
                if t.backends.mps.is_available()
                else "cpu"
            )


class Embed(nn.Module):
    """Token embedding. Parameter named W_E so state dict keys match the checkpoint."""

    def __init__(self, cfg: ShortformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.kaiming_uniform_(self.W_E, a=math.sqrt(5))

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    """Absolute learned positional embedding. Parameter named W_pos for key parity."""

    def __init__(self, cfg: ShortformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.n_ctx, cfg.d_model))
        nn.init.kaiming_uniform_(self.W_pos, a=math.sqrt(5))

    def forward(self, seq_len: int) -> Float[Tensor, "pos d_model"]:
        return self.W_pos[:seq_len]


class Unembed(nn.Module):
    """Unembedding. Parameters named W_U and b_U for key parity."""

    def __init__(self, cfg: ShortformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(cfg.d_model, cfg.d_vocab))
        self.b_U = nn.Parameter(t.zeros(cfg.d_vocab))
        nn.init.kaiming_uniform_(self.W_U, a=math.sqrt(5))

    def forward(
        self, residual: Float[Tensor, "batch pos d_model"]
    ) -> Float[Tensor, "batch pos d_vocab"]:
        return (
            einops.einsum(
                residual, self.W_U, "batch pos d_model, d_model d_vocab -> batch pos d_vocab"
            )
            + self.b_U
        )


class Attention(nn.Module):
    """Single-head-group attention layer with TransformerLens-style hooks.

    Shortformer semantics: queries and keys are computed from `resid + pos_embed`,
    values from `resid` alone.
    """

    def __init__(self, cfg: ShortformerConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.W_Q = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_K = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_V = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_O = nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.b_Q = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        self.b_O = nn.Parameter(t.zeros(cfg.d_model))
        for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

        # [n_ctx, n_ctx] with True iff query position can attend to key position
        # (query is first axis, key is second axis).
        self.register_buffer("mask", t.tril(t.ones(cfg.n_ctx, cfg.n_ctx, dtype=t.bool)))
        # Score substituted in for masked-out positions, before softmaxing.
        self.register_buffer("IGNORE", t.tensor(-t.inf))

        self.attn_scale = cfg.d_head**0.5

        self.hook_q = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_k = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_v = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_z = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, n_heads, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, n_heads, query_pos, key_pos]
        if cfg.use_attn_result:
            self.hook_result = HookPoint()  # [batch, pos, n_heads, d_model]

    def forward(
        self,
        resid: Float[Tensor, "batch pos d_model"],
        pos_embed: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        # Shortformer: pos_embed only enters the Q and K calculations.
        q = self.hook_q(
            einops.einsum(
                resid + pos_embed,
                self.W_Q,
                "batch pos d_model, head d_model d_head -> batch pos head d_head",
            )
            + self.b_Q
        )
        k = self.hook_k(
            einops.einsum(
                resid + pos_embed,
                self.W_K,
                "batch pos d_model, head d_model d_head -> batch pos head d_head",
            )
            + self.b_K
        )
        # Note: no pos_embed in the value calculation.
        v = self.hook_v(
            einops.einsum(
                resid, self.W_V, "batch pos d_model, head d_model d_head -> batch pos head d_head"
            )
            + self.b_V
        )

        attn_scores = (
            einops.einsum(
                q,
                k,
                "batch query_pos head d_head, batch key_pos head d_head -> batch head query_pos key_pos",
            )
            / self.attn_scale
        )
        query_pos, key_pos = attn_scores.shape[-2:]
        causal_mask = self.mask[:query_pos, :key_pos].bool()
        attn_scores = t.where(causal_mask, attn_scores, self.IGNORE)
        attn_scores = self.hook_attn_scores(attn_scores)

        pattern = self.hook_pattern(F.softmax(attn_scores, dim=-1))

        z = self.hook_z(
            einops.einsum(
                v,
                pattern,
                "batch key_pos head d_head, batch head query_pos key_pos -> batch query_pos head d_head",
            )
        )

        if self.cfg.use_attn_result:
            result = self.hook_result(
                einops.einsum(
                    z,
                    self.W_O,
                    "batch query_pos head d_head, head d_head d_model -> batch query_pos head d_model",
                )
            )
            attn_out = (
                einops.reduce(result, "batch pos head d_model -> batch pos d_model", "sum")
                + self.b_O
            )
        else:
            attn_out = (
                einops.einsum(
                    z,
                    self.W_O,
                    "batch query_pos head d_head, head d_head d_model -> batch query_pos d_model",
                )
                + self.b_O
            )
        return attn_out


class TransformerBlock(nn.Module):
    """Attention-only block, with TransformerLens-style residual hooks."""

    def __init__(self, cfg: ShortformerConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg, layer_id)
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[Tensor, "batch pos d_model"],
        pos_embed: Float[Tensor, "batch pos d_model"],
    ) -> Float[Tensor, "batch pos d_model"]:
        resid_pre = self.hook_resid_pre(resid_pre)
        attn_out = self.attn(resid_pre, pos_embed)
        resid_post = self.hook_resid_post(resid_pre + attn_out)
        return resid_post


class ShortformerTransformer(HookedRootModule):
    """Two-layer attention-only transformer with shortformer positional embeddings."""

    def __init__(self, cfg: ShortformerConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.seed is not None:
            t.manual_seed(cfg.seed)

        self.embed = Embed(cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        self.pos_embed = PosEmbed(cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_id=i) for i in range(cfg.n_layers)]
        )
        self.unembed = Unembed(cfg)

        self.tokenizer = None
        if cfg.tokenizer_name is not None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        self.to(cfg.device)
        self.setup()

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input: str | Int[Tensor, "batch pos"],
        return_type: Literal["logits", "loss", "both"] | None = "logits",
        loss_per_token: bool = False,
        prepend_bos: bool | None = None,
    ) -> Float[Tensor, "batch pos d_vocab"] | Tensor | tuple | None:
        """Forward pass. Input is a token tensor [batch, pos] or a text string.

        return_type="logits" | "loss" | "both" | None.
        """
        if isinstance(input, str):
            assert self.tokenizer is not None, "Must provide a tokenizer to pass strings."
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        if tokens.ndim == 1:
            tokens = tokens[None]
        tokens = tokens.to(self.cfg.device)

        seq_len = tokens.shape[1]
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos = self.hook_pos_embed(self.pos_embed(seq_len))[None]  # [1, pos, d_model]
        # Shortformer: positional embeddings are NOT added to the residual stream;
        # they're passed separately into each block's Q/K calculations.
        residual = embed

        for block in self.blocks:
            residual = block(residual, pos)

        if return_type is None:
            return None

        logits = self.unembed(residual)  # [batch, pos, d_vocab]
        if return_type == "logits":
            return logits

        # Cross-entropy of the shifted predictions.
        log_probs = logits.log_softmax(dim=-1)
        loss = -log_probs[:, :-1].gather(-1, tokens[:, 1:, None]).squeeze(-1)  # [batch, pos-1]
        if not loss_per_token:
            loss = loss.mean()
        if return_type == "loss":
            return loss
        return logits, loss

    # ------------------------------------------------- TransformerLens-style API

    def run_with_cache(self, *model_args, remove_batch_dim: bool = False, **kwargs):
        """As HookedRootModule.run_with_cache, but returns an ActivationCache."""
        model_out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=False, **kwargs
        )
        cache = ActivationCache(cache_dict, self)
        if remove_batch_dim:
            cache.remove_batch_dim()
        return model_out, cache

    # ------------------------------------------------------- weight accessors
    # Shapes follow TransformerLens conventions, e.g. model.W_Q[layer, head] has
    # shape [d_model, d_head] and model.W_O[layer, head] has [d_head, d_model].

    @property
    def W_E(self) -> Float[Tensor, "d_vocab d_model"]:
        return self.embed.W_E

    @property
    def W_U(self) -> Float[Tensor, "d_model d_vocab"]:
        return self.unembed.W_U

    @property
    def W_pos(self) -> Float[Tensor, "n_ctx d_model"]:
        return self.pos_embed.W_pos

    @property
    def W_Q(self) -> Float[Tensor, "n_layers n_heads d_model d_head"]:
        return t.stack([block.attn.W_Q for block in self.blocks])

    @property
    def W_K(self) -> Float[Tensor, "n_layers n_heads d_model d_head"]:
        return t.stack([block.attn.W_K for block in self.blocks])

    @property
    def W_V(self) -> Float[Tensor, "n_layers n_heads d_model d_head"]:
        return t.stack([block.attn.W_V for block in self.blocks])

    @property
    def W_O(self) -> Float[Tensor, "n_layers n_heads d_head d_model"]:
        return t.stack([block.attn.W_O for block in self.blocks])

    # ------------------------------------------------------ tokenizer utilities

    def to_tokens(
        self,
        input: str | list[str] | Int[Tensor, "batch pos"],
        prepend_bos: bool | None = None,
    ) -> Int[Tensor, "batch pos"]:
        """Convert text to a tensor of tokens (batch size 1 for a single string).

        Tensors are passed through unchanged (no prepending), matching
        HookedTransformer. Strings have a BOS token prepended by default.
        """
        if isinstance(input, t.Tensor):
            return input if input.ndim == 2 else input[None]
        assert self.tokenizer is not None, "Need a tokenizer (set cfg.tokenizer_name)."
        if isinstance(input, str):
            input = [input]
        prepend = self.cfg.default_prepend_bos if prepend_bos is None else prepend_bos
        bos = self.tokenizer.bos_token_id
        token_lists = []
        for text in input:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if prepend and bos is not None:
                ids = [bos] + ids
            token_lists.append(ids)
        max_len = max(len(ids) for ids in token_lists)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        return t.tensor([ids + [pad] * (max_len - len(ids)) for ids in token_lists], dtype=t.long)

    def to_str_tokens(
        self,
        input: str | Int[Tensor, "batch pos"],
        prepend_bos: bool | None = None,
    ):
        """Map text or tokens to a list of strings (one per token).

        For batched input, returns a list of such lists.
        """
        tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        assert self.tokenizer is not None
        # Decode each token id individually, like HookedTransformer does (rather than
        # convert_ids_to_tokens), so byte-level BPE artifacts are rendered as the actual
        # characters they encode: 'Ġ' -> space, 'Ċ' -> newline, etc. The per-token wrapping
        # is required with transformers v5, where batch_decode treats a flat list as a
        # single sequence rather than individual token ids.
        str_tokens = [
            self.tokenizer.batch_decode(
                [[int(token_id)] for token_id in row], clean_up_tokenization_spaces=False
            )
            for row in tokens.tolist()
        ]
        if isinstance(input, str) or (isinstance(input, t.Tensor) and input.ndim == 1):
            return str_tokens[0]
        return str_tokens

    def to_string(self, tokens: list | Int[Tensor, " pos"]) -> str:
        """Map token ids (list or tensor) back to a string."""
        assert self.tokenizer is not None
        if isinstance(tokens, t.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "callummcdougall/attn_only_2L_half",
        filename: str = "attn_only_2L_half.pth",
        cfg: ShortformerConfig | None = None,
        map_location: str = "cpu",
    ) -> ShortformerTransformer:
        """Download the pretrained weights from HuggingFace and load them."""
        from huggingface_hub import hf_hub_download

        if cfg is None:
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
        model = cls(cfg)
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = t.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)  # strict=True; keys match exactly
        return model
