import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from u2net import U2NET
import os # Added for path manipulation

# Load U2NET model
model_u2net = U2NET(3, 1) # Renamed to avoid conflict
model_path = 'checkpoints/u2net.pth'
# Ensure the model path is correct relative to the execution of test.py
# If unet.py and checkpoints/ are in the same directory as test.py, this is fine.
# script_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(script_dir, 'checkpoints', 'u2net.pth')

# Check if CUDA is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_u2net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model_u2net.to(device)
model_u2net.eval()

# Preprocessing function
def preprocess_image_for_unet(img_pil): # Renamed to be more specific
    transform = transforms.Compose([
        transforms.Resize((256, 192)), # Standard U2Net input size (height, width)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    return transform(img_pil).unsqueeze(0)

# Get mask from image
def get_unet_mask(image_pil, target_size): # Renamed, takes PIL image and target_size for mask
    input_tensor = preprocess_image_for_unet(image_pil)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        d1, *_ = model_u2net(input_tensor)
        pred = d1[0][0].cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min()) # Normalize to 0-1
        # Resize mask to target_size (W, H for cv2.resize)
        mask_resized = cv2.resize(pred, target_size, interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(mask_resized, axis=2) # (H, W, 1)

# Apply mask and change background to white
def apply_unet_mask_and_save(input_image_path, output_image_path):
    if not os.path.exists(input_image_path):
        print(f"[!] Lỗi: Không tìm thấy ảnh đầu vào tại: {input_image_path}")
        return

    original_pil = Image.open(input_image_path).convert('RGB')

    # VITON pipeline expects 192W x 256H.
    viton_target_size_pil = (192, 256) # (W, H) for PIL resize

    # Get U-Net mask, ask for it at original image's dimensions for accurate application
    mask = get_unet_mask(original_pil, target_size=(original_pil.width, original_pil.height))

    # Apply mask
    original_rgb_np = np.array(original_pil) # H, W, C
    # Ensure mask is broadcastable: (H, W, 1)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=2)
    # Double check resize if shapes don't match (should not happen if get_unet_mask is correct)
    if mask.shape[:2] != original_rgb_np.shape[:2]:
        mask = cv2.resize(mask, (original_rgb_np.shape[1], original_rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask.ndim == 2: mask = np.expand_dims(mask, axis=2)

    # Create white background image of the same size as original
    white_background = np.ones_like(original_rgb_np) * 255
    
    # Combine: foreground * mask + background * (1 - mask)
    mask_float = mask.astype(float) 
    result_rgb_np = (original_rgb_np * mask_float + white_background * (1 - mask_float)).astype(np.uint8)

    # Resize final result to VITON's expected dimensions (192W x 256H)
    result_pil = Image.fromarray(result_rgb_np)
    result_pil_resized = result_pil.resize(viton_target_size_pil, Image.LANCZOS)

    # Save
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    result_pil_resized.save(output_image_path)
    # print(f"[✔] Đã xử lý và lưu ảnh tại: {output_image_path}") # Moved to test.py for batch summary

# === CHẠY THỬ ===
if __name__ == '__main__':
    # Create dummy raw image for testing if it doesn't exist
    test_input_dir = "raw_images_unet_test"
    os.makedirs(test_input_dir, exist_ok=True)
    dummy_image_path = os.path.join(test_input_dir, "test_sample.jpg")
    
    if not os.path.exists(dummy_image_path):
        # Fallback: expect user to place an image or use a known test image
        if os.path.exists("test_2.jpg"): # From original context
            dummy_image_path = "test_2.jpg"
        else:
            print(f"Vui lòng đặt ảnh mẫu tại '{dummy_image_path}' hoặc 'test_2.jpg' để chạy thử unet.py.")
            exit()

    test_output_dir = "dataset/test_img_unet_output_standalone" 
    os.makedirs(test_output_dir, exist_ok=True)
    
    output_filename = os.path.basename(dummy_image_path)
    test_output_path = os.path.join(test_output_dir, output_filename)
    print(f"Chạy thử unet.py với đầu vào: {dummy_image_path}, đầu ra: {test_output_path}")
    apply_unet_mask_and_save(dummy_image_path, test_output_path)
    print(f"[✔] Chạy thử unet.py: Đã lưu ảnh tại: {test_output_path}")
