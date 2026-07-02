import torch
import torch.nn as nn
from torchvision.models.video import (
    r3d_18,      R3D_18_Weights,
    mc3_18,      MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
)

BACKBONE_MAP = {
    "r3d18":    (r3d_18,      R3D_18_Weights.DEFAULT),
    "mc318":    (mc3_18,      MC3_18_Weights.DEFAULT),
    "r2plus1d": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
}


def get_backbone(model_name):
    fn, weights = BACKBONE_MAP[model_name]
    base        = fn(weights=weights)
    img_dim     = base.fc.in_features
    backbone    = nn.Sequential(*list(base.children())[:-1])
    return backbone, img_dim


class ImageOnlyModel(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()

        self.backbone, img_dim = get_backbone(model_name)
        self.classifier        = nn.Linear(img_dim, 1)
        self.dropout           = nn.Dropout(dropout)

    def forward(self, img):
        feat = self.backbone(img).flatten(1)
        feat = self.dropout(feat)
        return self.classifier(feat)


class FusionModel(nn.Module):
    def __init__(self, model_name, dropout_image, dropout_tda, dropout_fusion, tda_dim=450):
        super().__init__()

        self.backbone, img_dim = get_backbone(model_name)

        self.image_head = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_image),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

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

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fusion),
            nn.Linear(64, 1),
        )

    def forward(self, img, tda):
        img_feat = self.backbone(img).flatten(1)
        img_feat = self.image_head(img_feat)
        tda_feat = self.tda_head(tda)
        fused    = torch.cat([img_feat, tda_feat], dim=1)
        return self.classifier(fused)
