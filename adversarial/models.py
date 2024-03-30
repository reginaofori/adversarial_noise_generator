import torchvision.models as models

def load_model(model_name='resnet18', pretrained=True):
    """
    Loads a pre-trained model from torchvision and set to evaluation mode.
    
    Parameters:
    - model_name (str): Name of the model to load. Default is 'resnet18'.
    - pretrained (bool): If True, loads a model pre-trained on ImageNet. Default is True.
    
    Returns:
    - A torchvision model in evaluation mode.
    """
    
    return getattr(models, model_name)(pretrained=pretrained).eval()
