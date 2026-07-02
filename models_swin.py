import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# ============================================================
# SWIN UNETR CONFIG
# ============================================================
FEATURE_SIZE = 48
IMG_SIZE     = (64, 64, 64)
IN_CHANNELS  = 3

# Deepest encoder output channels = feature_size × 16
ENCODER_DIM  = FEATURE_SIZE * 16   # 768


def load_swin_encoder(pretrained_weights_path):
    """
    Build SwinUNETR, load pretrained weights with 1→3 channel inflation,
    return encoder only (decoder discarded).
    """
    model = SwinUNETR(
        img_size     = IMG_SIZE,
        in_channels  = IN_CHANNELS,
        out_channels = 14,            # original segmentation classes — discarded
        feature_size = FEATURE_SIZE,
        use_checkpoint = True,        # gradient checkpointing — saves GPU memory
    )

    # load pretrained weights
    weights = torch.load(pretrained_weights_path, map_location="cpu")

    # pretrained weights have in_channels=1
    # inflate first layer: (C_out, 1, k, k, k) → (C_out, 3, k, k, k)
    state_dict = weights["state_dict"] if "state_dict" in weights else weights

    new_state_dict = {}

    for k, v in state_dict.items():
        # remove "module." prefix if present
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        # Inflate any 1-channel 3D convolution weight to 3 channels
        if (
            isinstance(v, torch.Tensor)
            and v.ndim == 5
            and v.shape[1] == 1
            and "weight" in k
        ):
            print(f"Inflating {k}: {tuple(v.shape)}", end=" -> ")
            v = v.repeat(1, IN_CHANNELS, 1, 1, 1) / IN_CHANNELS
            print(tuple(v.shape))

        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(
        new_state_dict,
        strict=False
    )
    print("Loaded Swin pretrained weights.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    # return only the Swin Transformer encoder
    return model.swinViT


class SwinImageOnlyModel(nn.Module):
    def __init__(self, pretrained_weights_path, dropout):
        super().__init__()

        self.encoder = load_swin_encoder(pretrained_weights_path)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ENCODER_DIM, 1),
        )

    def forward(self, img):
        # swinViT returns list of feature maps at 4 scales
        # we take the last (deepest) one
        features = self.encoder(img)
        x = features[-1]                  # (batch, 768, D', H', W')
        x = self.avgpool(x)               # (batch, 768, 1, 1, 1)
        return self.classifier(x)


class SwinFusionModel(nn.Module):
    def __init__(self, pretrained_weights_path, dropout_image, dropout_tda, dropout_fusion, tda_dim=450):
        super().__init__()

        self.encoder = load_swin_encoder(pretrained_weights_path)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # image head: 768 → 128 → 32
        self.image_head = nn.Sequential(
            nn.Linear(ENCODER_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout_image),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        # tda head: 450 → 128 → 64 → 32  (identical to CNN)
        self.tda_head = nn.Sequential(
            nn.Linear(tda_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_tda),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_tda),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # classifier: 64 → 1  (identical to CNN)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fusion),
            nn.Linear(64, 1),
        )

    def forward(self, img, tda):
        features = self.encoder(img)
        x        = features[-1]             # (batch, 768, D', H', W')
        x        = self.avgpool(x)          # (batch, 768, 1, 1, 1)
        x        = x.flatten(1)             # (batch, 768)

        img_feat = self.image_head(x)       # (batch, 32)
        tda_feat = self.tda_head(tda)       # (batch, 32)

        fused = torch.cat([img_feat, tda_feat], dim=1)  # (batch, 64)

        return self.classifier(fused)       # (batch, 1)
