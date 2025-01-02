import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def convert_to_rgb(img):
    """Check and convert the image to RGB if it's not in RGB format."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def augment_and_save_images(input_dir, output_dir, augment_count=200, img_size=(224, 224)):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomGrayscale(p=0.2),
    ])

    dataset = ImageFolder(root=input_dir, transform=transforms.Resize(img_size))

    for class_idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Augmenting images"):
        img = convert_to_rgb(img)  # Convert to RGB if necessary
        class_name = dataset.classes[label]
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        img_save_path = os.path.join(class_output_dir, f"original_{class_idx}.png")
        img.save(img_save_path)

        for i in range(augment_count):
            augmented_img = transform(img)
            aug_img_save_path = os.path.join(class_output_dir, f"augmented_{class_idx}_{i}.png")
            augmented_img.save(aug_img_save_path)

def copy_test_set(input_dir, output_dir, img_size=(224, 224)):
    dataset = ImageFolder(root=input_dir, transform=transforms.Resize(img_size))
    for class_idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Copying test images"):
        img = convert_to_rgb(img)  # Convert to RGB if necessary
        class_name = dataset.classes[label]
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        img_save_path = os.path.join(class_output_dir, f"test_{class_idx}.png")
        img.save(img_save_path)

if __name__ == "__main__":
    input_dir = "IFM_Dataset/copper_ring"
    output_dir = "IFM_Dataset/copper_ring_augmented"

    print("Starting data augmentation for the training set...")
    augment_and_save_images(input_dir=os.path.join(input_dir, "train"), output_dir=os.path.join(output_dir, "train"))

    print("Copying test set without augmentation...")
    copy_test_set(input_dir=os.path.join(input_dir, "test"), output_dir=os.path.join(output_dir, "test"))

    print("Data augmentation completed for the training set, and test set copied without augmentation in 'copper_ring_augmented'.")
