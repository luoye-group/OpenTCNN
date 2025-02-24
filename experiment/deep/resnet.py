import torchvision.models as pt_models

import tcnn.applications.vision.models as tcnn_models


def get_resnet34_models(num_classes=10):
    return {
        "resnet34": pt_models.resnet34(num_classes=num_classes),
        "type0_resnet34": tcnn_models.resnet34(num_classes=num_classes),
        "parallel_type1_resnet34": tcnn_models.parallel_type1_resnet34(
            num_classes=num_classes
        ),
        "parallel_type2_resnet34": tcnn_models.parallel_type2_resnet34(
            num_classes=num_classes
        ),
        "compound_type1_resnet34": tcnn_models.compound_type1_resnet34(
            num_classes=num_classes
        ),
        "compound_type2_resnet34": tcnn_models.compound_type2_resnet34(
            num_classes=num_classes
        ),
    }


def get_resnet18_models(num_classes=10):
    return {
        "parallel_type1_resnet18": tcnn_models.parallel_type1_resnet18(
            num_classes=num_classes
        ),
        "parallel_type2_resnet18": tcnn_models.parallel_type2_resnet18(
            num_classes=num_classes
        ),
        "compound_type1_resnet18": tcnn_models.compound_type1_resnet18(
            num_classes=num_classes
        ),
        "compound_type2_resnet18": tcnn_models.compound_type2_resnet18(
            num_classes=num_classes
        ),
        "resnet18": pt_models.resnet18(num_classes=num_classes),
        "type0_resnet18": tcnn_models.resnet18(num_classes=num_classes),
    }
