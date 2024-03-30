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
    # Enable gradient calculation for the image
    image.requires_grad = True  

    # Calculate loss to obtain image gradient
    loss = torch.nn.functional.cross_entropy(model(image), target)
    model.zero_grad()
    loss.backward()

    # Generate and apply perturbation
    perturbation = epsilon * image.grad.sign()
    perturbed_image = (image + perturbation).clamp(0, 1)

    return perturbed_image
