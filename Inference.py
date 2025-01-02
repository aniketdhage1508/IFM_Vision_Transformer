import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS
import matplotlib.cm as cm  # For colormap
import matplotlib

# Labels
labels = {0: "not_present", 1: "present"}

# Convert image to RGB if not already
def convert_to_rgb(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# Load model
def load_model(model_path):
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=2, zero_head=False, img_size=224, vis=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=False)
    model.eval()  # Set to evaluation mode
    return model

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Visualize heatmap
def visualize_attention(model, image_path, output_path):
    im = Image.open(image_path)
    im = convert_to_rgb(im)

    x = transform(im).unsqueeze(0)  # Add batch dimension
    logits, att_mat = model(x)

    # Process attention matrix
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = mask / mask.max()  # Normalize

    # Resize and overlay heatmap
    mask = cv2.resize(mask, im.size)[..., np.newaxis]
    heatmap = matplotlib.colormaps['jet'](mask.squeeze())[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)
    im = np.array(im)
    heatmap_overlay = cv2.addWeighted(heatmap, 0.5, im, 0.5, 0)

    # Display predictions
    probs = torch.nn.Softmax(dim=-1)(logits)
    top_preds = torch.argsort(probs, dim=-1, descending=True)[0, :5]

    # Add prediction text to the heatmap image
    text_start_y = 30  # Starting Y-coordinate for text
    font_scale = 0.6
    font_color = (255, 255, 255)  # White text
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in top_preds:
        label_text = f'{probs[0, idx.item()]:.5f} : {labels[idx.item()]}'
        cv2.putText(
            heatmap_overlay, 
            label_text, 
            (10, text_start_y), 
            font, 
            font_scale, 
            font_color, 
            thickness, 
            lineType=cv2.LINE_AA
        )
        text_start_y += 20  # Increment Y-coordinate for next line of text

    # Save the heatmap image
    cv2.imwrite(output_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))

    print("Prediction Label and Attention Map:")
    for idx in top_preds:
        print(f'{probs[0, idx.item()]:.5f} : {labels[idx.item()]}')

# Main function
if __name__ == "__main__":
    model_path = "output/washer_5000_B_16_Aug200_checkpoint_Final.bin"
    image_path = "testing/washer_recognition_washer_0031.png"
    output_path = "results/Heatmap.png"

    model = load_model(model_path)
    visualize_attention(model, image_path, output_path)

    print(f"Heatmap saved to {output_path}.")
