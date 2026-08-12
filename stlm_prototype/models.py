"""
Models for the pixel and description-substrate STLM pilots.

The descriptive component names used by new code are:

    SP    substrate predictor
    TR-S  single-state token readout
    TR-H  history-conditioned token readout
    TR-C  contextual token readout (causal sequence model)

The original class names remain available so published checkpoints and old
scripts continue to load unchanged.
"""
import torch
import torch.nn.functional as F
from torch import nn


class ImageClassifier(nn.Module):
    """Pixel TR-S. CNN on a 3xHxW substrate state -> vocabulary logits."""

    def __init__(self, img_size, n_tokens, channels=(32, 64, 128, 256),
                 fc_hidden=256):
        super().__init__()
        layers, in_c = [], 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c),
                nn.ReLU(), nn.MaxPool2d(2)]
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        reduced = img_size // (2 ** len(channels))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c * reduced * reduced, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, n_tokens))

    def forward(self, img):                    # img: [B, 3, H, W]
        return self.head(self.conv(img))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ImagePredictor(nn.Module):
    """Pixel SP. Causal transformer -> per-position 3xHxW pixel image."""

    def __init__(self, n_tokens, img_size, img_ch, context_len,
                 txt_layers=6, txt_d_model=384, txt_heads=6):
        super().__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.context_len = context_len
        self.tok_emb = nn.Embedding(n_tokens, txt_d_model)
        self.pos_emb = nn.Embedding(context_len, txt_d_model)
        block = nn.TransformerEncoderLayer(
            txt_d_model, txt_heads, dim_feedforward=4 * txt_d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(
            block, txt_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(txt_d_model)
        self.head = nn.Linear(txt_d_model, img_ch * img_size * img_size)

    def _encode(self, ids):
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=ids.device)
        x = self.blocks(x, mask=mask)
        return self.norm(x)

    def _project(self, x):
        flat = torch.sigmoid(self.head(x))
        return flat.view(*flat.shape[:-1], self.img_ch,
                         self.img_size, self.img_size)

    def forward(self, ids):
        """ids: [B, T] -> [B, T, 3, H, W] in [0, 1]."""
        return self._project(self._encode(ids))

    def forward_last(self, ids):
        """Project only the final context position for readout training."""
        return self._project(self._encode(ids)[:, -1])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class DescriptionEmbeddingPredictor(nn.Module):
    """Description SP. Causal transformer -> unit-norm description embeddings.

    For each context position, emits a sequence of `desc_len` unit-norm
    vectors -- the predicted per-subword BERT hidden states of the next
    word's description. The shared head + per-slot positional embedding
    pattern keeps parameter count tractable: one linear projection per
    slot would be desc_len x large, so we add a per-slot position embedding
    to the body output and project through one shared linear head.

    L2-normalization at the seam closes off SP's output norm as a private
    side channel a token readout could otherwise exploit. Only direction
    crosses the interface; cosine distance is the natural loss.
    """

    def __init__(self, n_tokens, desc_len, embed_dim, context_len,
                 txt_layers=6, txt_d_model=384, txt_heads=6):
        super().__init__()
        self.desc_len = desc_len
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.tok_emb = nn.Embedding(n_tokens, txt_d_model)
        self.pos_emb = nn.Embedding(context_len, txt_d_model)
        self.desc_pos_emb = nn.Embedding(desc_len, txt_d_model)
        block = nn.TransformerEncoderLayer(
            txt_d_model, txt_heads, dim_feedforward=4 * txt_d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(
            block, txt_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(txt_d_model)
        self.desc_norm = nn.LayerNorm(txt_d_model)
        self.desc_head = nn.Linear(txt_d_model, embed_dim)

    def _encode(self, ids):
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=ids.device)
        x = self.blocks(x, mask=mask)
        return self.norm(x)

    def _project(self, x):
        desc_pos = torch.arange(self.desc_len, device=x.device)
        desc = x[..., None, :] + self.desc_pos_emb(desc_pos)
        desc = self.desc_norm(desc)
        return F.normalize(self.desc_head(desc), dim=-1)

    def forward(self, ids):
        """ids: [B, T] -> [B, T, desc_len, embed_dim], unit-norm."""
        return self._project(self._encode(ids))

    def forward_last(self, ids):
        """Project only the final context position for readout training."""
        return self._project(self._encode(ids)[:, -1])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class DescriptionEmbeddingClassifier(nn.Module):
    """Description TR-S. Transformer reader for SP's predicted state.

    TR-S sees only the [desc_len, embed_dim] substrate emitted by SP. It
    receives no story context tokens and no canonical attention mask. PAD
    positions are part of the substrate itself (a fixed vector direction),
    so the reader must learn from vector content which positions matter.
    """

    def __init__(self, embed_dim, n_tokens, desc_len,
                 d_model=384, n_heads=6, n_layers=1, ff_dim=1536):
        super().__init__()
        self.embed_dim = embed_dim
        self.desc_len = desc_len
        self.d_model = d_model
        self.input_proj = (
            nn.Identity() if embed_dim == d_model
            else nn.Linear(embed_dim, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Embedding(desc_len + 1, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_dim,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerEncoder(
            block, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_tokens)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, emb):
        """emb: [B, L, embed_dim] -> [B, n_tokens]."""
        B, L, _ = emb.shape
        if L != self.desc_len:
            raise ValueError(f"expected desc_len={self.desc_len}, got {L}")
        x = self.input_proj(emb)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(L + 1, device=emb.device)
        x = x + self.pos_emb(pos)[None]
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class _CausalSubstrateSequenceModel(nn.Module):
    """Decoder-only Transformer over already-encoded substrate states."""

    def __init__(
            self, d_model, n_tokens, max_context, n_heads, n_layers, ff_dim):
        super().__init__()
        if max_context < 1:
            raise ValueError("max_context must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.max_context = max_context
        self.pos_emb = nn.Embedding(max_context, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, states):
        """Map ``[B, T, d_model]`` states to per-position token logits."""
        if states.ndim != 3 or states.shape[-1] != self.d_model:
            raise ValueError(
                f"expected states [batch, time, {self.d_model}]")
        time = states.shape[1]
        if not 1 <= time <= self.max_context:
            raise ValueError(
                f"sequence length must be in [1, {self.max_context}], "
                f"got {time}")
        positions = torch.arange(time, device=states.device)
        hidden = states + self.pos_emb(positions)[None]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            time, device=states.device)
        hidden = self.blocks(hidden, mask=causal_mask)
        return self.head(self.norm(hidden))


class PixelSubstrateStateEncoder(nn.Module):
    """Lightweight, per-state pixel adapter with no cross-time statistics."""

    def __init__(self, img_size, d_model=256, pool_grid=8):
        super().__init__()
        if pool_grid < 1:
            raise ValueError("pool_grid must be positive")
        self.img_size = img_size
        self.d_model = d_model
        self.pool_grid = pool_grid
        pooled_dim = 3 * pool_grid * pool_grid
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_grid, pool_grid)),
            nn.Flatten(),
            nn.Linear(pooled_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, states):
        if states.ndim != 4:
            raise ValueError("pixel states must have shape [N, 3, H, W]")
        if tuple(states.shape[1:]) != (
                3, self.img_size, self.img_size):
            raise ValueError(
                f"expected pixel states [3, {self.img_size}, "
                f"{self.img_size}], got {list(states.shape[1:])}")
        return self.adapter(states)


class DescriptionSubstrateStateEncoder(nn.Module):
    """Pool ordered description regions into one per-state feature."""

    def __init__(self, embed_dim, desc_len, d_model=256, pool_bins=8):
        super().__init__()
        if pool_bins < 1:
            raise ValueError("pool_bins must be positive")
        self.embed_dim = embed_dim
        self.desc_len = desc_len
        self.d_model = d_model
        self.pool_bins = pool_bins
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim * pool_bins, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, states):
        if states.ndim != 3:
            raise ValueError(
                "description states must have shape [N, desc_len, embed_dim]")
        _, length, embed_dim = states.shape
        if length != self.desc_len or embed_dim != self.embed_dim:
            raise ValueError(
                f"expected description states [{self.desc_len}, "
                f"{self.embed_dim}], got [{length}, {embed_dim}]")
        pooled = F.adaptive_avg_pool1d(
            states.transpose(1, 2), self.pool_bins)
        # Restore the sequence axis before flattening so each contiguous block
        # is one ordered pooled region of the description substrate.
        flattened = pooled.transpose(1, 2).reshape(states.shape[0], -1)
        return self.adapter(flattened)


class PixelContextualTokenReadout(nn.Module):
    """TR-C: causal token model over a complete pixel-substrate sequence."""

    def __init__(
            self, img_size, n_tokens, context_len, d_model=256,
            n_heads=8, n_layers=2, ff_dim=1024, pool_grid=8):
        super().__init__()
        self.img_size = img_size
        self.context_len = context_len
        self.d_model = d_model
        self.state_encoder = PixelSubstrateStateEncoder(
            img_size=img_size,
            d_model=d_model,
            pool_grid=pool_grid,
        )
        self.sequence_model = _CausalSubstrateSequenceModel(
            d_model=d_model,
            n_tokens=n_tokens,
            max_context=context_len,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
        )

    def forward(self, trace):
        """``trace: [B,T,3,H,W] -> [B,T,n_tokens]``."""
        if trace.ndim != 5:
            raise ValueError(
                "pixel trace must have shape [batch, time, 3, H, W]")
        batch, time, channels, height, width = trace.shape
        flat = trace.reshape(batch * time, channels, height, width)
        states = self.state_encoder(flat).reshape(batch, time, self.d_model)
        return self.sequence_model(states)

    def n_params(self):
        return sum(parameter.numel() for parameter in self.parameters())


class DescriptionContextualTokenReadout(nn.Module):
    """TR-C: causal token model over a complete description-state sequence."""

    def __init__(
            self, embed_dim, desc_len, n_tokens, context_len,
            d_model=256, temporal_heads=8, temporal_layers=2,
            temporal_ff_dim=1024, pool_bins=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.desc_len = desc_len
        self.context_len = context_len
        self.d_model = d_model
        self.state_encoder = DescriptionSubstrateStateEncoder(
            embed_dim=embed_dim,
            desc_len=desc_len,
            d_model=d_model,
            pool_bins=pool_bins,
        )
        self.sequence_model = _CausalSubstrateSequenceModel(
            d_model=d_model,
            n_tokens=n_tokens,
            max_context=context_len,
            n_heads=temporal_heads,
            n_layers=temporal_layers,
            ff_dim=temporal_ff_dim,
        )

    def forward(self, trace):
        """``trace: [B,T,L,D] -> [B,T,n_tokens]``."""
        if trace.ndim != 4:
            raise ValueError(
                "description trace must have shape [batch, time, L, D]")
        batch, time, length, embed_dim = trace.shape
        flat = trace.reshape(batch * time, length, embed_dim)
        states = self.state_encoder(flat).reshape(batch, time, self.d_model)
        return self.sequence_model(states)

    def n_params(self):
        return sum(parameter.numel() for parameter in self.parameters())


class _HistoryResidualAggregator(nn.Module):
    """Aggregate lightweight per-state features into residual token logits."""

    def __init__(
            self, state_dim, ff_dim, n_tokens, max_history, n_heads=4):
        super().__init__()
        if max_history < 1:
            raise ValueError("max_history must be positive")
        if state_dim % n_heads:
            raise ValueError("state_dim must be divisible by n_heads")
        self.max_history = max_history
        self.cls = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.pos_emb = nn.Embedding(max_history + 1, state_dim)
        block = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            block,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(state_dim)
        self.head = nn.Linear(state_dim, n_tokens)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # A newly constructed TR-H initially reproduces frozen TR-S exactly.
        # The first update learns the residual head; gradients then reach the
        # history encoder and temporal encoder on subsequent updates.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, states):
        if states.ndim != 3:
            raise ValueError(
                "history features must have shape [batch, time, feature]")
        if states.shape[1] == 0:
            return states.new_zeros((states.shape[0], self.head.out_features))
        batch, time, _ = states.shape
        if time > self.max_history:
            raise ValueError(
                f"history length {time} exceeds configured maximum "
                f"{self.max_history}")
        cls = self.cls.expand(batch, -1, -1)
        sequence = torch.cat([cls, states], dim=1)
        positions = torch.arange(time + 1, device=states.device)
        sequence = sequence + self.pos_emb(positions)[None]
        encoded = self.temporal(sequence)
        return self.head(self.norm(encoded[:, 0]))


class _FrozenSingleStateMixin:
    """Keep the published TR-S branch frozen and in inference mode."""

    def _freeze_single_state_readout(self):
        self.single_state_readout.train(False)
        for parameter in self.single_state_readout.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # `Module.train()` recurses. Restore inference mode so pixel TR-S
        # BatchNorm statistics cannot drift during TR-H training.
        self.single_state_readout.train(False)
        return self

    def total_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_params(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class PixelHistoryConditionedTokenReadout(
        _FrozenSingleStateMixin, nn.Module):
    """TR-H for a causal trace of pixel-substrate states.

    The current state follows the exact published pixel TR-S path. Earlier
    states are spatially pooled and passed through a small shared projection
    and temporal Transformer encoder; that branch contributes residual logits.
    The readout receives only substrate states, never token IDs or text.
    """

    def __init__(
            self,
            img_size,
            n_tokens,
            current_channels=(32, 64, 128, 256),
            current_fc_hidden=256,
            history_grid=8,
            history_state_dim=128,
            history_ff_dim=256,
            history_heads=4,
            max_history=127):
        super().__init__()
        if history_grid < 1:
            raise ValueError("history_grid must be positive")
        self.img_size = img_size
        self.n_tokens = n_tokens
        self.history_grid = history_grid
        self.history_state_dim = history_state_dim
        self.history_ff_dim = history_ff_dim
        self.history_heads = history_heads
        self.max_history = max_history

        self.single_state_readout = ImageClassifier(
            img_size=img_size,
            n_tokens=n_tokens,
            channels=current_channels,
            fc_hidden=current_fc_hidden,
        )
        pooled_dim = 3 * history_grid * history_grid
        self.history_state_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((history_grid, history_grid)),
            nn.Flatten(),
            nn.Linear(pooled_dim, history_state_dim),
            nn.GELU(),
            nn.LayerNorm(history_state_dim),
        )
        self.history_aggregator = _HistoryResidualAggregator(
            state_dim=history_state_dim,
            ff_dim=history_ff_dim,
            n_tokens=n_tokens,
            max_history=max_history,
            n_heads=history_heads,
        )
        self._freeze_single_state_readout()

    def load_single_state_readout(self, state_dict):
        self.single_state_readout.load_state_dict(state_dict)
        self._freeze_single_state_readout()

    def current_logits(self, current):
        """Read one current pixel state through the frozen TR-S."""
        return self.single_state_readout(current)

    def history_logits(self, history):
        """Return the learned residual contributed by prior pixel states."""
        if history.ndim != 5:
            raise ValueError(
                "pixel history must have shape [batch, time, 3, H, W]")
        batch, time, channels, height, width = history.shape
        if channels != 3 or height != self.img_size or width != self.img_size:
            raise ValueError(
                f"expected pixel states [3, {self.img_size}, {self.img_size}], "
                f"got [{channels}, {height}, {width}]")
        if time == 0:
            return history.new_zeros((batch, self.n_tokens))
        flat = history.reshape(batch * time, channels, height, width)
        features = self.history_state_encoder(flat)
        features = features.reshape(batch, time, self.history_state_dim)
        return self.history_aggregator(features)

    def forward(self, trace, include_current=True, include_history=True):
        """Read ``[B, K, 3, H, W]``; the final state is current."""
        if trace.ndim != 5 or trace.shape[1] < 1:
            raise ValueError(
                "pixel trace must have shape [batch, time>=1, 3, H, W]")
        batch = trace.shape[0]
        logits = trace.new_zeros((batch, self.n_tokens))
        if include_current:
            logits = logits + self.current_logits(trace[:, -1])
        if include_history and trace.shape[1] > 1:
            logits = logits + self.history_logits(trace[:, :-1])
        return logits


class DescriptionHistoryConditionedTokenReadout(
        _FrozenSingleStateMixin, nn.Module):
    """TR-H for a causal trace of description-substrate states.

    The frozen current-state path is the exact published description TR-S.
    For history, adaptive pooling retains a small ordered set of description
    regions before a shared projection and temporal Transformer encoder produce
    residual token logits. No token IDs, text, or canonical masks enter this
    module.
    """

    def __init__(
            self,
            embed_dim,
            n_tokens,
            desc_len,
            current_d_model=384,
            current_n_heads=6,
            current_n_layers=1,
            current_ff_dim=1536,
            history_bins=8,
            history_state_dim=128,
            history_ff_dim=256,
            history_heads=4,
            max_history=127):
        super().__init__()
        if history_bins < 1:
            raise ValueError("history_bins must be positive")
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.desc_len = desc_len
        self.history_bins = history_bins
        self.history_state_dim = history_state_dim
        self.history_ff_dim = history_ff_dim
        self.history_heads = history_heads
        self.max_history = max_history

        self.single_state_readout = DescriptionEmbeddingClassifier(
            embed_dim=embed_dim,
            n_tokens=n_tokens,
            desc_len=desc_len,
            d_model=current_d_model,
            n_heads=current_n_heads,
            n_layers=current_n_layers,
            ff_dim=current_ff_dim,
        )
        pooled_dim = embed_dim * history_bins
        self.history_state_encoder = nn.Sequential(
            nn.Linear(pooled_dim, history_state_dim),
            nn.GELU(),
            nn.LayerNorm(history_state_dim),
        )
        self.history_aggregator = _HistoryResidualAggregator(
            state_dim=history_state_dim,
            ff_dim=history_ff_dim,
            n_tokens=n_tokens,
            max_history=max_history,
            n_heads=history_heads,
        )
        self._freeze_single_state_readout()

    def load_single_state_readout(self, state_dict):
        self.single_state_readout.load_state_dict(state_dict)
        self._freeze_single_state_readout()

    def current_logits(self, current):
        """Read one current description state through the frozen TR-S."""
        return self.single_state_readout(current)

    def _pool_history_states(self, history):
        batch, time, length, embed_dim = history.shape
        flat = history.reshape(batch * time, length, embed_dim)
        # Pool along the ordered description-position axis. Keeping multiple
        # bins retains coarse order/binding at far lower cost than applying
        # the full TR-S transformer independently to every history state.
        pooled = F.adaptive_avg_pool1d(
            flat.transpose(1, 2), self.history_bins)
        return pooled.reshape(batch * time, embed_dim * self.history_bins)

    def history_logits(self, history):
        """Return the learned residual contributed by prior descriptions."""
        if history.ndim != 4:
            raise ValueError(
                "description history must have shape "
                "[batch, time, desc_len, embed_dim]")
        batch, time, length, embed_dim = history.shape
        if length != self.desc_len or embed_dim != self.embed_dim:
            raise ValueError(
                f"expected description states [{self.desc_len}, "
                f"{self.embed_dim}], got [{length}, {embed_dim}]")
        if time == 0:
            return history.new_zeros((batch, self.n_tokens))
        pooled = self._pool_history_states(history)
        features = self.history_state_encoder(pooled)
        features = features.reshape(batch, time, self.history_state_dim)
        return self.history_aggregator(features)

    def forward(self, trace, include_current=True, include_history=True):
        """Read ``[B, K, desc_len, embed_dim]``; final state is current."""
        if trace.ndim != 4 or trace.shape[1] < 1:
            raise ValueError(
                "description trace must have shape "
                "[batch, time>=1, desc_len, embed_dim]")
        batch = trace.shape[0]
        logits = trace.new_zeros((batch, self.n_tokens))
        if include_current:
            logits = logits + self.current_logits(trace[:, -1])
        if include_history and trace.shape[1] > 1:
            logits = logits + self.history_logits(trace[:, :-1])
        return logits


# Descriptive aliases for new code. Original names above remain canonical for
# checkpoint compatibility with the two published pilot runs.
PixelSubstratePredictor = ImagePredictor
DescriptionSubstratePredictor = DescriptionEmbeddingPredictor
PixelSingleStateTokenReadout = ImageClassifier
DescriptionSingleStateTokenReadout = DescriptionEmbeddingClassifier
