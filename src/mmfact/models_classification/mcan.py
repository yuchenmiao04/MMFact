"""MCAN: spatial/frequency image features and repeated bidirectional co-attention."""

import math

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder, _PretrainedEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


class DCTEncoder(_PretrainedEncoder):
    """Fixed 8-by-8 block-frequency histograms, Fourier transformed and sampled.

    The paper refers to Qi et al. (2019) for this transform. Histogram bins,
    orthonormal DCT and deterministic sampling are specified in the recipe;
    the released notebook's transforms over a singleton axis are not copied.
    """

    input_kind = "image"
    packages = ("torch", "numpy", "Pillow", "scipy")

    def __init__(self, image_size=224):
        super().__init__()
        if image_size < 8 or image_size % 8:
            raise ValueError("MCAN frequency image_size must be a positive multiple of 8.")
        self.image_size = image_size
        self.network = nn.Identity()
        self.encoding_spec = {
            "encoder": "mcan_dct_histogram",
            "version": 1,
            "image_size": image_size,
            "color": "PIL L",
            "resize": "bilinear",
            "pixel_shift": -128,
            "block_size": 8,
            "dct": "type_2_ortho_2d",
            "histogram": "round_to_integer_2048_bins_-1024_to_1023_probability",
            "fourier": "fft_real",
            "sampling": "nearest_250",
        }
        self._set_frozen(True)

    @staticmethod
    def _histograms(pixels):
        from scipy.fft import dctn

        height, width = pixels.shape
        blocks = (pixels.astype(np.float32) - 128).reshape(height // 8, 8, width // 8, 8)
        blocks = blocks.transpose(0, 2, 1, 3)
        coefficients = dctn(blocks, type=2, axes=(-2, -1), norm="ortho")
        frequencies = coefficients.reshape(-1, 64).T
        bins = np.rint(frequencies).astype(np.int64).clip(-1024, 1023) + 1024
        histograms = np.stack([np.bincount(row, minlength=2048) for row in bins])
        return histograms.astype(np.float32) / frequencies.shape[1]

    def _encode(self, paths):
        results = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                histograms = self._histograms(np.asarray(image))
            spectrum = torch.from_numpy(np.fft.fft(histograms, axis=-1).real.copy()).float()
            results.append(F.interpolate(spectrum.unsqueeze(0), size=250, mode="nearest")[0])
        return torch.stack(results).to(self.device)


def _conv(in_channels, out_channels, width, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (1, width), padding=(0, padding)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class _FrequencyCNN(nn.Module):
    """Figure 3: shared CNN over 64 frequencies, with four Inception branches."""

    output_dim = 4096

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            _conv(1, 32, 3), _conv(32, 64, 3), _conv(64, 128, 3), nn.MaxPool2d((1, 2))
        )
        self.branches = nn.ModuleList(
            [
                _conv(128, 64, 1),
                nn.Sequential(_conv(128, 48, 1), _conv(48, 64, 3, 1)),
                nn.Sequential(_conv(128, 64, 1), _conv(64, 96, 3, 1), _conv(96, 96, 3, 1)),
                nn.Sequential(nn.MaxPool2d((1, 3), stride=1, padding=(0, 1)), _conv(128, 32, 1)),
            ]
        )
        self.output = nn.Sequential(nn.MaxPool2d((1, 122)), _conv(256, 64, 1))

    def forward(self, frequencies):
        features = self.stem(frequencies.unsqueeze(1))
        features = self.output(torch.cat([branch(features) for branch in self.branches], dim=1))
        return features.transpose(1, 2).flatten(1)


class _Projection(nn.Module):
    """Author release's linear/ReLU/dropout/BN, including singleton batches."""

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.norm = SingletonBatchNorm(hidden_dim)

    def forward(self, values):
        return self.norm(self.layers(values))


class _CoAttentionBlock(nn.Module):
    """Equations 1-7: scalar feature coordinates are the attention positions."""

    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        if dimension < 1 or heads < 1 or dimension % heads:
            raise ValueError("MCAN hidden_dim must be positive and divisible by num_heads.")
        self.heads, self.head_dim = heads, dimension // heads
        self.query = nn.Linear(1, dimension, bias=False)
        self.key = nn.Linear(1, dimension, bias=False)
        self.value = nn.Linear(1, dimension, bias=False)
        self.output = nn.Linear(dimension, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dimension)
        )
        self.ffn_norm = nn.LayerNorm(dimension)

    def _heads(self, values, projection):
        projected = projection(values.unsqueeze(-1))
        return projected.reshape(*values.shape, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, context):
        q, k, v = (
            self._heads(values, projection)
            for values, projection in (
                (query, self.query),
                (context, self.key),
                (context, self.value),
            )
        )
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        attended = (self.dropout(weights) @ v).transpose(1, 2).flatten(2)
        values = self.attention_norm(query + self.dropout(self.output(attended).squeeze(-1)))
        return self.ffn_norm(values + self.dropout(self.ffn(values)))


class _CoAttentionLayer(nn.Module):
    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        self.left = _CoAttentionBlock(dimension, heads, ffn_dim, dropout)
        self.right = _CoAttentionBlock(dimension, heads, ffn_dim, dropout)
        self.fusion = nn.Linear(2 * dimension, dimension)

    def forward(self, left, right):
        return self.fusion(torch.cat((self.left(left, right), self.right(right, left)), dim=-1))


@register_model("MCAN")
class MCAN(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        dimension, dropout = parameters["hidden_dim"], parameters["dropout"]
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        self.frequency_encoder = material_encoder(
            config, "frequency_encoder", DCTEncoder, **parameters["frequency"]
        )
        self.frequency_cnn = _FrequencyCNN()
        self.text_projection = _Projection(self.text_encoder.output_dim, dimension, dropout)
        self.image_projection = _Projection(self.image_encoder.output_dim, dimension, dropout)
        self.frequency_projection = _Projection(self.frequency_cnn.output_dim, dimension, dropout)
        self.fusion_layers = nn.ModuleList(
            [
                _CoAttentionLayer(
                    dimension, parameters["num_heads"], parameters["ffn_dim"], dropout
                )
                for _ in range(2)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(dimension, parameters["classifier_hidden"]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(parameters["classifier_hidden"], 2),
        )

    def forward(self, batch):
        text = self.text_projection(self.text_encoder(batch["text"]))
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        frequency = self.frequency_projection(
            self.frequency_cnn(self.frequency_encoder(batch["image_path"]))
        )
        for context in (frequency, text):
            for layer in self.fusion_layers:
                image = layer(image, context)
        return self.classifier(image)

    def calculate_loss(self, batch):
        return F.cross_entropy(self(batch), batch["labels"])

    def predict(self, batch):
        return self(batch)
