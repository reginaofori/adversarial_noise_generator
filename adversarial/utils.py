from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, size=(224, 224)):

    """
    Preprocess an image file to a format suitable for model input.

    Parameters:
    - image_path: str, path to the image file.
    - size: tuple, target size of the image (height, width).

    Returns:
    - image_tensor: torch.Tensor, preprocessed image tensor.
    """
    preprocessing = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image_tensor = preprocessing(image).unsqueeze(0)  
    return image_tensor

def postprocess_image(image_tensor):
    """
    Postprocess a torch.Tensor image to a PIL image.

    Parameters:
    - image_tensor: torch.Tensor, the image tensor to postprocess.

    Returns:
    - image: PIL.Image, the postprocessed image.
    """
    image_tensor = image_tensor.squeeze(0)  
    image = transforms.ToPILImage()(image_tensor)
    return image
