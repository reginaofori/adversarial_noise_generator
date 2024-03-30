import torchvision.models as models

def load_model(model_name='resnet18', pretrained=True):
    """
    Load a pre-trained model from torchvision's available models.

    Parameters:
    - model_name: str, name of the model to load.
    - pretrained: bool, if True, loads a model pre-trained on ImageNet.

    Returns:
    - model: torchvision model, the loaded pre-trained model.
    """
    model = getattr(models, model_name)(pretrained=pretrained)
    # Setting the model to evaluation mode
    model.eval()  
    return model
