import torch

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import ProjectionEmbedding
from mmf.utils.build import build_classifier_layer, build_image_encoder


@registry.register_model("concat_vl")
class LanguageAndVisionConcat(BaseModel):
    # Not really needed, you can skip this as it doesn't do anything
    # Left here for the brevity purposes
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    @classmethod
    def config_path(cls):
        # Relative to user dir root
        return "configs/models/concat_vl.yaml"
    
    def build(self):
        self.vision_module = build_image_encoder(self.config.image_encoder)
        self.classifier = build_classifier_layer(self.config.classifier)
        self.language_module = ProjectionEmbedding(**self.config.text_encoder.params)
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.fusion = torch.nn.Linear(**self.config.fusion.params)

    def forward(self, sample_list):
        text = sample_list["text"]
        image = sample_list["image"]
        text_features = torch.nn.functional.relu(
            self.language_module(text)
        )
        image_features = torch.nn.functional.relu(
            self.vision_module(image)
        )
        combined = torch.cat(
            [text_features, image_features.squeeze(dim=1)], dim=1
        )
        fused = self.dropout(
            torch.nn.functional.relu(
                self.fusion(combined)
            )
        )
        logits = self.classifier(fused)

        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output