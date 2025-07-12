import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm 


try:
    from unet import apply_unet_mask_and_save, model_u2net as unet_model_loaded # Ensure U-Net model is loaded
except ImportError as e:
    print(f"Không thể import từ unet.py: {e}")
    print("Hãy đảm bảo unet.py và các file phụ thuộc (như u2net.py) nằm trong PYTHONPATH hoặc cùng thư mục.")
    exit(1)

def get_expected_person_image_basenames(opt):
    """
    Reads the pairing file (e.g., test_pairs.txt) to get the list of person image basenames
    that AlignedDataset will expect.
    """
    # Default phase is 'test' for TestOptions
    # AlignedDataset typically uses opt.phase + '_pairs.txt' or a fixed name.
    # For CP-VTON and derivatives, 'test_pairs.txt' is common.
    pairs_file_name = getattr(opt, 'pairs_file', 'test_pairs.txt') # Allow overriding if needed
    pairs_file_path = os.path.join(opt.dataroot, pairs_file_name)

    if not os.path.exists(pairs_file_path):
        print(f"[!] Lỗi: Không tìm thấy tệp danh sách cặp ảnh '{pairs_file_path}'.")
        print(f"    Tệp này cần thiết để xác định ảnh người mẫu nào cần xử lý qua U-Net.")
        print(f"    Vui lòng tạo tệp này trong '{opt.dataroot}' với mỗi dòng chứa tên ảnh người và tên ảnh áo, ví dụ: 'person_image.jpg cloth_image.jpg'.")
        return None

    person_basenames = []
    with open(pairs_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1: # Expecting at least the person image name
                person_basenames.append(parts[0])
            else:
                print(f"[!] Cảnh báo: Dòng không hợp lệ trong '{pairs_file_path}': '{line.strip()}'")
    
    if not person_basenames:
        print(f"[!] Không tìm thấy tên ảnh người mẫu nào trong '{pairs_file_path}'.")
        return None
        
    return person_basenames

def preprocess_selected_images_with_unet(raw_image_dir, processed_image_dir, image_basenames_to_process):
    """
    Processes specified images from raw_image_dir using U-Net and saves them to processed_image_dir
    with their original basenames.
    """
    if not os.path.exists(raw_image_dir):
        print(f"[!] Thư mục ảnh gốc '{raw_image_dir}' không tồn tại.")
        print(f"    Vui lòng tạo thư mục '{os.path.abspath(raw_image_dir)}' và đặt các ảnh: {', '.join(image_basenames_to_process)} vào đó.")
        return False

    os.makedirs(processed_image_dir, exist_ok=True)
    print(f"[*] Bắt đầu xử lý ảnh bằng U-Net từ '{os.path.abspath(raw_image_dir)}' sang '{os.path.abspath(processed_image_dir)}'...")

    processed_count = 0
    missing_raw_files = []

    for basename in tqdm(image_basenames_to_process, desc="Xử lý ảnh qua U-Net"):
        input_path = os.path.join(raw_image_dir, basename)
        output_path = os.path.join(processed_image_dir, basename)

        if not os.path.exists(input_path):
            missing_raw_files.append(basename)
            continue
        
        try:
            apply_unet_mask_and_save(input_path, output_path)
            processed_count += 1
        except Exception as e:
            print(f"[!] Lỗi khi xử lý ảnh '{input_path}' bằng U-Net: {e}")
    
    if missing_raw_files:
        print(f"[!] Cảnh báo: Không tìm thấy các tệp ảnh gốc sau trong '{raw_image_dir}': {', '.join(missing_raw_files)}")
        print(f"    Hãy đảm bảo các tệp này tồn tại trong '{raw_image_dir}' với tên chính xác như trong tệp danh sách cặp ảnh (ví dụ: test_pairs.txt).")

    if processed_count > 0:
        print(f"[✔] Đã xử lý xong {processed_count} ảnh bằng U-Net.")
    else:
        print(f"[!] Không có ảnh nào được xử lý. Kiểm tra thư mục '{raw_image_dir}' và tệp danh sách cặp ảnh.")
        return False
        
    return True


if __name__ == '__main__':
    opt = TestOptions().parse()

    # --- BEGIN U-NET PREPROCESSING ---
    RAW_IMAGE_INPUT_DIR = 'raw_images' # Tạo thư mục này và đặt ảnh gốc vào đây
    # Thư mục này sẽ được DataLoader sử dụng, ví dụ: 'dataset/test_img'
    # Đảm bảo đường dẫn này khớp với cách CreateDataLoader tìm ảnh 'image'.
    if not opt.dataroot:
        print("[!] Lỗi: opt.dataroot không được thiết lập. Không thể xác định thư mục dataset.")
        exit(1)
    
    TARGET_PERSON_IMG_DIR = os.path.join(opt.dataroot, 'test_img')

    print(f"[*] Thư mục ảnh gốc cho U-Net: {os.path.abspath(RAW_IMAGE_INPUT_DIR)}")
    print(f"[*] Thư mục ảnh đã xử lý (đầu vào cho VITON): {os.path.abspath(TARGET_PERSON_IMG_DIR)}")

    # Get the list of person image basenames that AlignedDataset will expect
    expected_person_basenames = get_expected_person_image_basenames(opt)

    if not expected_person_basenames:
        print("[!] Không thể xác định danh sách ảnh người mẫu dự kiến từ tệp cặp ảnh. Dừng chương trình.")
        exit(1)
    
    print(f"[*] Các ảnh người mẫu dự kiến sẽ được xử lý (từ tệp cặp ảnh): {', '.join(expected_person_basenames)}")

    success_unet_processing = preprocess_selected_images_with_unet(
        RAW_IMAGE_INPUT_DIR,
        TARGET_PERSON_IMG_DIR,
        expected_person_basenames
    )

    if not success_unet_processing:
        print("[!] Xử lý U-Net không thành công hoặc không có ảnh nào được xử lý. Dừng chương trình.")
        exit(1)

    if not os.path.exists(TARGET_PERSON_IMG_DIR) or not os.listdir(TARGET_PERSON_IMG_DIR):
        print(f"[!] Thư mục '{TARGET_PERSON_IMG_DIR}' trống sau khi xử lý U-Net (hoặc xử lý thất bại/không có ảnh gốc).")
        print(f"    Vui lòng đảm bảo các ảnh thô tương ứng với danh sách trong tệp cặp ảnh tồn tại trong '{os.path.abspath(RAW_IMAGE_INPUT_DIR)}' và U-Net xử lý thành công.")
        exit(1)
    # --- END U-NET PREPROCESSING ---

    start_epoch, epoch_iter = 1, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f"[*] Kích thước dataset (sau xử lý U-Net): {dataset_size}")
    if dataset_size == 0:
        print(f"[!] Không có dữ liệu nào được tải bởi DataLoader từ '{TARGET_PERSON_IMG_DIR}'. Kiểm tra cấu hình DataLoader và đường dẫn.")
        exit(1)

    warp_model = AFWM(opt, 3)
    # print(warp_model) # Verbose
    warp_model.eval()
    warp_model.cuda()
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    # print(gen_model) # Verbose
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint(gen_model, opt.gen_checkpoint)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    step = 0
    # step_per_batch = dataset_size / opt.batchSize # Not used

    results_base_path = 'results' 
    os.makedirs(results_base_path, exist_ok=True)

    print("[*] Bắt đầu vòng lặp thử đồ ảo...")
    for epoch in range(1,2):

        for i, data in enumerate(dataset, start=epoch_iter):
            # iter_start_time = time.time() # Not used
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            real_image = data['image'].cuda() 
            clothes = data['clothes'].cuda()
            
            edge = data['edge'].cuda() 
            if edge.dtype != torch.float32:
                 edge = edge.float()
            edge_binary = (edge > 0.5).float() 
            
            clothes = clothes * edge_binary 

            flow_out = warp_model(real_image, clothes) 
            warped_cloth, last_flow = flow_out
            warped_edge = F.grid_sample(edge_binary, last_flow.permute(0, 2, 3, 1),
                            mode='bilinear', padding_mode='zeros')

            gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
            gen_outputs = gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)


            if step % 1 == 0: 
                img_a_vis = real_image[0].cpu().detach()

                img_b_vis = data['clothes'][0].cpu().detach() 
                img_c_vis = p_tryon[0].cpu().detach()

                combine = torch.cat([img_a_vis, img_b_vis, img_c_vis], 2).squeeze() 
                cv_img = (combine.permute(1, 2, 0).numpy() + 1) / 2 
                rgb_img = (cv_img * 255).astype(np.uint8)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
                output_image_filename = f"tryon_step_{step:04d}.jpg"
                output_image_path = os.path.join(results_base_path, output_image_filename)
                cv2.imwrite(output_image_path, bgr_img)
                if step % 10 == 0: 
                    print(f"[*] Đã lưu ảnh kết quả: {output_image_path}")

            step += 1
            if epoch_iter >= dataset_size:
                break
        epoch_iter = 0 
    print(f"[✔] Hoàn thành xử lý. Kết quả được lưu trong thư mục: {os.path.abspath(results_base_path)}")
