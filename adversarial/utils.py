from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, size=(224, 224)):
    """
    Preprocesses an image file for model input, resizing it and converting to a tensor.

    Parameters:
    - image_path (str): Path to the image file.
    - size (tuple): Target size of the image (height, width), default is (224, 224).

    Returns:
    - torch.Tensor: Preprocessed image tensor with an added batch dimension.
    """
    # Defines the preprocessing pipeline
    preprocessing = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    # Opens the image, apply preprocessing, and add a batch dimension
    image = Image.open(image_path)
    return preprocessing(image).unsqueeze(0)  



def postprocess_image(image_tensor):
    """
    Converts a torch.Tensor back into a PIL image, removing the batch dimension.

    Parameters:
    - image_tensor (torch.Tensor): The image tensor to convert.

    Returns:
    - PIL.Image: The postprocessed image.
    """
    # Convert the tensor to a PIL image after removing the batch dimension
    return transforms.ToPILImage()(image_tensor.squeeze(0))
