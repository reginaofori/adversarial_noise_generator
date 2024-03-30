import argparse
import os
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
from adversarial.models import load_model
from adversarial.attacks import apply_fgsm_attack
from adversarial.utils import preprocess_image, postprocess_image



def load_class_labels(json_filepath):
    with open(json_filepath, 'r') as file:
        class_labels = json.load(file)
    return class_labels


def validate_image_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified image path does not exist: {path}")

def get_class_index(class_name_or_index):
    if class_name_or_index.isdigit():
        return int(class_name_or_index)
    else:
        class_name = class_name_or_index.lower().replace(" ", "_")
        class_labels = load_class_labels('target_class.json')

        if class_name in class_labels:
            return class_labels[class_name]
        else:
            raise ValueError(f"Class name '{class_name_or_index}' not recognized.")

def visualize_images(original_image_path, adversarial_image):
    original_image = Image.open(original_image_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(adversarial_image)
    axs[1].set_title('Adversarial Image')
    axs[1].axis('off')

    plt.show()

def main(image_path, target_class, epsilon, model_name):
    validate_image_path(image_path)
    target_class_index = get_class_index(target_class)

    # Load a pre-trained model
    model = load_model(model_name, pretrained=True)

    # Preprocess the input image
    image_tensor = preprocess_image(image_path)

    # Specifying the target class
    target = torch.tensor([target_class_index], dtype=torch.long)

    # ApplyING FGSM attack
    adversarial_image_tensor = apply_fgsm_attack(image_tensor, target, model, epsilon)

    # Postprocessing and saving the adversarial image
    adversarial_image = postprocess_image(adversarial_image_tensor)
    adversarial_image.save('adversarial_image.jpg')

    print("Adversarial image generated and saved as 'adversarial_image.jpg'")
    visualize_images(image_path, adversarial_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an adversarial image using FGSM attack.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("target_class", help="Target class name or index for the adversarial image")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Perturbation magnitude (default: 0.05)")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name to use (default: resnet18)")
    args = parser.parse_args()

    try:
        main(args.image_path, args.target_class, args.epsilon, args.model_name)
    except Exception as e:
        print(f"Error: {e}")
