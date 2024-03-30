import torch
from adversarial.models import load_model
from adversarial.attacks import apply_fgsm_attack
from adversarial.utils import preprocess_image

def test_fgsm_attack():
    model = load_model('resnet18', pretrained=True)
    image_tensor = torch.rand(1, 3, 224, 224)  
    target = torch.tensor([208], dtype=torch.long)  
    epsilon = 0.1

    adversarial_image_tensor = apply_fgsm_attack(image_tensor, target, model, epsilon)
    assert adversarial_image_tensor is not None, "FGSM attack failed to generate an adversarial image"
    assert adversarial_image_tensor.shape == image_tensor.shape, "Adversarial image has incorrect dimensions"
