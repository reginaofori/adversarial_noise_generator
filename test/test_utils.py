import os
import torch
from adversarial.utils import preprocess_image, postprocess_image


def test_preprocess_image():
    test_images_directory = 'test/images/'  # Ensure this directory exists and contains your test images
    for filename in os.listdir(test_images_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image formats as needed
            image_path = os.path.join(test_images_directory, filename)
            image_tensor = preprocess_image(image_path)
            assert isinstance(image_tensor, torch.Tensor), f"Preprocessing {filename} should return a torch.Tensor"



def test_postprocess_image_various_sizes():
    sizes = [(224, 224), (256, 256), (128, 128)]  # Example sizes
    for size in sizes:
        image_tensor = torch.rand(1, 3, *size)  
        image = postprocess_image(image_tensor)
        assert image.size == size[::-1], f"Postprocessing should return an image of size {size[::-1]}"
