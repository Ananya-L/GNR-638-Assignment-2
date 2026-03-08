import timm
import torch.nn as nn


def load_model(model_name, num_classes=30, pretrained=True):

    model = timm.create_model(
        model_name,
        pretrained=pretrained
    )

    # replace classifier
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model

def freeze_backbone(model):

    for name, param in model.named_parameters():

        if "fc" not in name and "classifier" not in name:
            param.requires_grad = False