"""MVAE: a joint variational representation with text/image reconstruction.

Khattar et al., WWW 2019, https://doi.org/10.1145/3308558.3313552.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import WordEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


def gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    """KL(q(z|x) || N(0, I)), summed over latent dimensions and batch-averaged."""
    return 0.5 * (mean.square() + log_variance.exp() - log_variance - 1).sum(-1).mean()


def _kernel_penalty(module: nn.Module) -> Tensor:
    # Keras kernel_regularizer excludes biases and LSTM recurrent kernels.
    return sum(
        parameter.square().sum()
        for name, parameter in module.named_parameters()
        if name.endswith("weight") or name.startswith("weight_ih")
    )


@register_model("MVAE")
class MVAE(AbstractDetector):
    """Frozen Word2Vec/ResNet inputs, a bimodal VAE, and a binary detector."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        self.text = material_encoder(config, "text", WordEncoder, **parameters["text"])
        self.image = material_encoder(config, "image", supervised_encoder, **parameters["image"])
        hidden = parameters.get("hidden_dim", 32)
        image_hidden = parameters.get("image_hidden_dim", 1024)
        latent = parameters.get("latent_dim", 64)
        self.autoencoder_l2 = parameters.get("autoencoder_l2", 0.05)
        self.classifier_l2 = parameters.get("classifier_l2", 0.3)
        self.sampling_std = parameters.get("sampling_std", 1.0)
        self.kl_reduction = parameters.get("kl_reduction", "sum")
        self.text_reconstruction_reduction = parameters.get(
            "text_reconstruction_reduction", "sum_per_sample"
        )
        if self.text_reconstruction_reduction not in {"sum_per_sample", "mean_per_token"}:
            raise ValueError(
                "MVAE text_reconstruction_reduction must be sum_per_sample or mean_per_token"
            )
        if (
            isinstance(self.sampling_std, bool)
            or not isinstance(self.sampling_std, (int, float))
            or not math.isfinite(self.sampling_std)
            or self.sampling_std <= 0
        ):
            raise ValueError("MVAE sampling_std must be finite and positive")
        if self.kl_reduction not in {"sum", "mean"}:
            raise ValueError("MVAE kl_reduction must be sum or mean")

        self.text_encoder = nn.LSTM(
            self.text.dimension, hidden, num_layers=2, bidirectional=True, batch_first=True
        )
        self.text_projection = nn.Linear(2 * hidden, hidden)
        self.image_encoder = nn.Sequential(
            nn.Linear(self.image.output_dim, image_hidden),
            nn.Tanh(),
            nn.Linear(image_hidden, hidden),
            nn.Tanh(),
        )
        self.shared = nn.Linear(2 * hidden, latent)
        self.mean = nn.Linear(latent, latent)
        self.log_variance = nn.Linear(latent, latent)

        self.text_decoder_input = nn.Linear(latent, hidden)
        self.text_decoder = nn.LSTM(
            hidden, hidden, num_layers=2, bidirectional=True, batch_first=True
        )
        self.word_decoder = nn.Linear(2 * hidden, self.text.max_vocab_size)
        self.image_decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.Tanh(),
            nn.Linear(hidden, image_hidden),
            nn.Tanh(),
            nn.Linear(image_hidden, self.image.output_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent, latent),
            nn.Tanh(),
            nn.Linear(latent, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def prepare(self, train_dataset) -> None:
        self.text.prepare([sample["text"] for sample in train_dataset])

    def _encode(self, batch: Mapping[str, Any]) -> tuple[Tensor, ...]:
        embeddings, tokens, mask = self.text(batch["text"])
        lengths = mask.sum(-1).clamp_min(1).cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (states, _) = self.text_encoder(packed)
        textual = torch.tanh(self.text_projection(torch.cat((states[-2], states[-1]), dim=-1)))
        image_features = self.image(batch["image_path"])
        visual = self.image_encoder(image_features)
        shared = torch.tanh(self.shared(torch.cat((textual, visual), dim=-1)))
        return self.mean(shared), self.log_variance(shared), tokens, mask, image_features

    def _decode_text(self, latent: Tensor, mask: Tensor) -> Tensor:
        context = torch.tanh(self.text_decoder_input(latent))
        repeated = context.unsqueeze(1).expand(-1, mask.size(1), -1)
        lengths = mask.sum(-1).clamp_min(1).cpu()
        packed = pack_padded_sequence(repeated, lengths, batch_first=True, enforce_sorted=False)
        decoded, _ = self.text_decoder(packed)
        decoded, _ = pad_packed_sequence(decoded, batch_first=True, total_length=mask.size(1))
        # The train-only vocabulary may be smaller than its allocated capacity.
        size = self.text.vocab_size
        return F.linear(
            decoded[mask], self.word_decoder.weight[:size], self.word_decoder.bias[:size]
        )

    def _regularization(self) -> dict[str, Tensor]:
        autoencoder = sum(
            _kernel_penalty(module)
            for module in (
                self.text_encoder,
                self.text_projection,
                self.image_encoder,
                self.shared,
                self.text_decoder_input,
                self.text_decoder,
                self.image_decoder[:-1],
            )
        )
        return {
            "autoencoder_l2": self.autoencoder_l2 * autoencoder,
            "classifier_l2": self.classifier_l2 * _kernel_penalty(self.classifier[:-1]),
        }

    def calculate_loss(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        mean, log_variance, tokens, mask, image_features = self._encode(batch)
        # Sampling and KL must describe the same posterior, including any noise scale.
        effective_log_variance = log_variance + 2.0 * math.log(self.sampling_std)
        latent = mean + (0.5 * effective_log_variance).exp() * torch.randn_like(mean)
        words = self._decode_text(latent, mask)
        return {
            "classification": F.binary_cross_entropy_with_logits(
                self.classifier(latent).squeeze(-1), batch["labels"].float()
            ),
            "text_reconstruction": F.cross_entropy(words, tokens[mask], reduction="sum")
            / (
                mask.sum().clamp_min(1)
                if self.text_reconstruction_reduction == "mean_per_token"
                else tokens.size(0)
            ),
            "image_reconstruction": F.mse_loss(self.image_decoder(latent), image_features),
            "kl": gaussian_kl(mean, effective_log_variance)
            / (mean.size(-1) if self.kl_reduction == "mean" else 1),
            **self._regularization(),
        }

    def predict(self, batch: Mapping[str, Any]) -> Tensor:
        # Posterior-mean inference gives stable validation and test predictions.
        mean, *_ = self._encode(batch)
        fake = self.classifier(mean)
        return torch.cat((torch.zeros_like(fake), fake), dim=-1)
