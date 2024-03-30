from adversarial.models import load_model

def test_load_model():
    model = load_model('resnet18', pretrained=True)
    assert model is not None, "Failed to load the pre-trained ResNet18 model"
