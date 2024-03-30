import torch

def apply_fgsm_attack(image, target, model, epsilon):
    """
    Apply the Fast Gradient Sign Method (FGSM) attack to generate an adversarial image.

    Parameters:
    - image: torch.Tensor, input image tensor.
    - target: torch.Tensor, target class for the adversarial example.
    - model: torchvision model, pre-trained model to fool.
    - epsilon: float, magnitude of the perturbation.

    Returns:
    - perturbed_image: torch.Tensor, image tensor with adversarial noise applied.
    """
    image.requires_grad = True  # Enable gradient calculation for the image

    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    image_grad = image.grad.data

    # Generate adversarial image by applying the perturbation
    perturbation = epsilon * image_grad.sign()
    perturbed_image = image + perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1) 

    return perturbed_image
