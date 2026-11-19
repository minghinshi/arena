# %% Setup
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn.functional as F
import trackio
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tokenize_utils import tokenize_and_concatenate

# These imports no longer work due to library updates
# from transformer_lens import HookedTransformer
# from transformer_lens.utils import gelu_new, tokenize_and_concatenate
# from transformers import GPT2TokenizerFast

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part1_transformer_from_scratch import tests

MAIN = __name__ == "__main__"

# --- Part 1 ---

# %% Load gpt2-small
reference_gpt2 = TransformerBridge.boot_transformers("gpt2")
reference_gpt2.enable_compatibility_mode(
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
)

# %% Check its tokenizer
sorted_vocab = sorted(reference_gpt2.tokenizer.vocab.items(), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()

# %% Tokenize
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %% Forward pass
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)

# %% Softmax
probs = logits.softmax(dim=-1)
print(probs.shape)

# %% Inference (all tokens)
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(
    logits.argmax(dim=-1)[0].unsqueeze(-1)
)

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# %% Inference (final token)
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

# %% Repeat inference for multiple tokens
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1] + 1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)

# --- Part 2 ---

# %% Reference model activation shapes
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")


# %% Reference model parameter shapes
for name, param in reference_gpt2.state_dict().items():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:30} {tuple(param.shape)}")


# %% Reference model config
print(reference_gpt2.cfg)


# %% Define GPT-2 config
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)


# %% Exercise 2.1: LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        mean = t.mean(residual, dim=-1, keepdim=True)
        var = t.var(residual, dim=-1, keepdim=True, unbiased=False)
        var = var + self.cfg.layer_norm_eps

        return (residual - mean) / t.sqrt(var) * self.w + self.b


tests.rand_float_test(LayerNorm, [2, 4, 768])
tests.test_layer_norm_epsilon(LayerNorm, cache["resid_post", 11])


# %% Exercise 2.2: Embed
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:

        # Haha advanced indexing goes brrrr
        return self.W_E[tokens]


tests.test_embed(Embed)
tests.rand_int_test(Embed, [2, 4])


# %% Exercise 2.3: PosEmbed
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:

        n_batch, seq_len = tokens.shape
        ids = t.arange(0, seq_len).repeat((n_batch, 1))
        return self.W_pos[ids]


tests.test_pos_embed(PosEmbed)
tests.rand_int_test(PosEmbed, [2, 4])


# %% Exercise 2.4 to 2.5: Attention
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros(cfg.d_model))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)

        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        normalized_resid_pre = einops.repeat(
            normalized_resid_pre,
            "b p d -> b h p d",
            h=self.cfg.n_heads,
        )

        # Calculate QKV matrices
        # Shape: batch n_heads pos d_head
        Q = normalized_resid_pre @ self.W_Q
        Q = Q + einops.rearrange(self.b_Q, "h d -> h 1 d")

        K = normalized_resid_pre @ self.W_K
        K = K + einops.rearrange(self.b_K, "h d -> h 1 d")

        V = normalized_resid_pre @ self.W_V
        V = V + einops.rearrange(self.b_V, "h d -> h 1 d")

        # Calculate attention weights
        # Shape: batch n_heads pos pos
        attn_scores = Q @ t.transpose(K, -2, -1)
        attn_scores = attn_scores / self.cfg.d_head**0.5
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention weights
        # Shape: batch n_heads pos d_head
        V = attn_probs @ V

        # Project up
        # Shape: batch n_heads pos d_model
        O = V @ self.W_O

        # Sum heads and add bias
        O = einops.reduce(O, "b h p d -> b p d", "sum")
        O = O + self.b_O
        return O

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        _, _, n_query, n_key = attn_scores.shape
        # If for whatever reason the attention matrix is not square,
        # then we have a problem
        assert n_query == n_key

        mask = t.ones((n_query, n_key), device=attn_scores.device)
        mask = t.triu(mask, diagonal=1) == 1
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)
tests.rand_float_test(Attention, [2, 4, 768])


# %% Exercise 2.6: MLP
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        x = normalized_resid_mid @ self.W_in + self.b_in
        x = F.gelu(x, approximate="tanh")
        x = x @ self.W_out + self.b_out
        return x


tests.rand_float_test(MLP, [2, 4, 768])


# %% Exercise 2.7: TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid = resid_pre.clone()
        resid = resid + self.attn(self.ln1(resid))
        resid = resid + self.mlp(self.ln2(resid))
        return resid


tests.rand_float_test(TransformerBlock, [2, 4, 768])


# %% Exercise 2.8: Unembed
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return normalized_resid_final @ self.W_U + self.b_U


tests.test_unembed(Unembed)
tests.rand_float_test(Unembed, [2, 4, 768])


# %% Exercise 2.9: DemoTransformer (the entire GPT-2)
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # Calculate embeddings (initializes the residual stream)
        resid = self.embed(tokens) + self.pos_embed(tokens)

        # Transformer blocks
        for block in self.blocks:
            resid = block(resid)

        # Unembed to get token distribution
        resid = self.ln_final(resid)
        return self.unembed(resid)


tests.rand_int_test(DemoTransformer, [2, 4])


# %% Load weights from GPT-2 to our homemade version
def get_compatible_state_dict():
    sd = reference_gpt2.state_dict()
    n_layers, n_heads = cfg.n_layers, cfg.n_heads

    new_sd = {
        "embed.W_E": sd["embed.weight"],
        "pos_embed.W_pos": sd["pos_embed.weight"],
        "ln_final.w": sd["ln_final.weight"],
        "ln_final.b": sd["ln_final.bias"],
        "unembed.W_U": sd["unembed.weight"].T,  # HF lm_head is [d_vocab, d_model]
        "unembed.b_U": sd["unembed.bias"],
    }

    for i in range(n_layers):
        prefix = f"blocks.{i}."

        new_sd[prefix + "ln1.w"] = sd[prefix + "ln1.weight"]
        new_sd[prefix + "ln1.b"] = sd[prefix + "ln1.bias"]
        new_sd[prefix + "ln2.w"] = sd[prefix + "ln2.weight"]
        new_sd[prefix + "ln2.b"] = sd[prefix + "ln2.bias"]

        for layer_name in ("q", "k", "v"):
            new_sd[prefix + f"attn.W_{layer_name.upper()}"] = einops.rearrange(
                sd[prefix + f"attn.{layer_name}.weight"],
                "(n h) d_model -> n d_model h",
                n=n_heads,
            )
            new_sd[prefix + f"attn.b_{layer_name.upper()}"] = einops.rearrange(
                sd[prefix + f"attn.{layer_name}.bias"],
                "(n h) -> n h",
                n=n_heads,
            )

        new_sd[prefix + "attn.W_O"] = einops.rearrange(
            sd[prefix + "attn.o.weight"],
            "(n h) d_model -> n h d_model",
            n=n_heads,
        )
        new_sd[prefix + "attn.b_O"] = sd[prefix + "attn.o.bias"]

        new_sd[prefix + "mlp.W_in"] = sd[prefix + "mlp.in.weight"]
        new_sd[prefix + "mlp.b_in"] = sd[prefix + "mlp.in.bias"]
        new_sd[prefix + "mlp.W_out"] = sd[prefix + "mlp.out.weight"]
        new_sd[prefix + "mlp.b_out"] = sd[prefix + "mlp.out.bias"]

    return new_sd


demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
result = demo_gpt2.load_state_dict(get_compatible_state_dict(), strict=False)
assert result.unexpected_keys == []

demo_logits = demo_gpt2(tokens)


# %% Calculate cross-entropy loss of example
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs for the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %% Greedily generate text
test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""

for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

# --- Part 3 ---

# %% Create a small transformer
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)


# %% Define training args
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str = "day1-demotransformer"
    wandb_name: str | None = None
    eval_prompt: str = "Once upon a time"


args = TransformerTrainingArgs()

# %% Load and preprocess dataset
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# Keep the first 200k of the 2.1M stories, as we only need a subset for training.
dataset = dataset.select(range(200_000))

print(dataset)
print(dataset[0]["text"])

tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=8,
)
assert isinstance(tokenized_dataset, datasets.Dataset)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)

train_loader = DataLoader(
    dataset_dict["train"],  # type: ignore
    batch_size=args.batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    dataset_dict["test"],  # type: ignore
    batch_size=args.batch_size,
    shuffle=False,
)

# %% Inspect training data
first_batch = train_loader.dataset[: args.batch_size]

print(first_batch.keys())
print(first_batch["tokens"].shape)


# %% Exercise 3.1: TransformerTrainer
# TODO: Delete this
import solutions


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = solutions.TransformerSampler(self.model, reference_gpt2.tokenizer)  # type: ignore
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],  # type: ignore
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],  # type: ignore
            batch_size=args.batch_size,
            shuffle=False,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch["tokens"]
        tokens = tokens.to(device)

        logits = model(tokens)
        loss = -get_log_probs(logits, tokens).mean()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1
        trackio.log({"loss": loss.item()}, self.step)

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        n_tokens = 0
        n_correct = 0

        for batch in self.test_loader:
            tokens: Tensor = batch["tokens"]
            tokens = tokens.to(device)

            logits = model(tokens)
            labels = t.argmax(logits, -1)
            is_correct = labels[:, :-1] == tokens[:, 1:]

            n_tokens += t.numel(is_correct)
            n_correct += t.sum(is_correct).item()

        accuracy = n_correct / n_tokens
        trackio.log({"accuracy": accuracy}, self.step)

        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        trackio.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=asdict(self.args),
        )
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))

        trackio.finish()


model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()
