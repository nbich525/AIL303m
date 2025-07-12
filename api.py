import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import time
from contextlib import asynccontextmanager


try:
    from options.test_options import TestOptions
    from models.afwm import AFWM 
    from models.networks import ResUnetGenerator
    from models.networks import load_checkpoint
    from unet import apply_unet_mask_and_save, model_u2net as unet_model_global # Model U-Net được tải sẵn từ unet.py
except ImportError as e:
    print(f"Lỗi import: {e}. Hãy đảm bảo PYTHONPATH được thiết lập đúng hoặc api.py nằm ở vị trí có thể thấy các module này.")
    raise


class AppState:
    opt = None
    warp_model = None
    gen_model = None

    device = None

app_state = AppState()

def get_img_transform(target_size=(256, 192), normalize=True): 
    transform_list = [transforms.Resize(target_size, interpolation=Image.BICUBIC)]
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

def get_edge_transform(target_size=(256, 192)): 
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=Image.NEAREST),
        transforms.ToTensor() 
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Đang tải models và cấu hình...")

    app_state.opt = TestOptions().parse([]) 
    

    app_state.opt.gpu_ids = [0] if torch.cuda.is_available() else []
    app_state.opt.batchSize = 1
    app_state.opt.isTrain = False
    app_state.opt.serial_batches = True 
    

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    app_state.opt.dataroot = os.path.join(current_file_dir, "dataset")
    print(f"Thiết lập dataroot thành: {app_state.opt.dataroot}")

    if not os.path.isdir(app_state.opt.dataroot):
        print(f"CẢNH BÁO: Thư mục dataroot '{app_state.opt.dataroot}' không tồn tại. Các đường dẫn đến ảnh áo/biên có thể sai.")

    app_state.device = torch.device(f"cuda:{app_state.opt.gpu_ids[0]}" if app_state.opt.gpu_ids and torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {app_state.device}")

 
    unet_model_global.to(app_state.device)


    app_state.warp_model = AFWM(app_state.opt, 3) 
    load_checkpoint(app_state.warp_model, app_state.opt.warp_checkpoint)
    app_state.warp_model.to(app_state.device)
    app_state.warp_model.eval()


    app_state.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    load_checkpoint(app_state.gen_model, app_state.opt.gen_checkpoint)
    app_state.gen_model.to(app_state.device)
    app_state.gen_model.eval()
    
    print("Models đã được tải thành công.")
    yield

    print("Đang tắt API.")

app = FastAPI(lifespan=lifespan)

# --- Thiết lập thư mục ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_TEMP_DIR = os.path.join(BASE_DIR, "api_temp_files")
RAW_PERSON_UPLOAD_DIR = os.path.join(API_TEMP_DIR, "raw_person_uploads")
UNET_PERSON_OUTPUT_DIR = os.path.join(API_TEMP_DIR, "unet_person_outputs")
FINAL_RESULTS_DIR = os.path.join(API_TEMP_DIR, "final_results")


CLOTHES_DIR_IN_DATAROOT = "test_clothes"
CLOTHES_EDGE_DIR_IN_DATAROOT = "test_edge"

for d in [API_TEMP_DIR, RAW_PERSON_UPLOAD_DIR, UNET_PERSON_OUTPUT_DIR, FINAL_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def perform_viton_tryon_logic(
    opt_config, raw_person_image_path, cloth_basename,
    unet_output_dir, final_result_dir,
    warp_model_loaded, gen_model_loaded, unet_model_instance, current_device
):
    request_id = str(uuid.uuid4())

    raw_person_filename = os.path.basename(raw_person_image_path)

    safe_raw_person_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in raw_person_filename)
    unet_processed_person_filename = f"{request_id}_{safe_raw_person_filename}"
    unet_processed_person_path = os.path.join(unet_output_dir, unet_processed_person_filename)

    # 1. Xử lý ảnh người qua U-Net
    try:
  
        apply_unet_mask_and_save(raw_person_image_path, unet_processed_person_path)
    except Exception as e:
        import traceback
        print(f"Lỗi U-Net processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh người qua U-Net: {str(e)}")

    if not os.path.exists(unet_processed_person_path):
        raise HTTPException(status_code=500, detail="Không tìm thấy output của U-Net.")

    person_pil = Image.open(unet_processed_person_path).convert('RGB')
    person_transform = transforms.Compose([ # U-Net đã resize, chỉ cần ToTensor và Normalize
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    I_tensor = person_transform(person_pil).unsqueeze(0).to(current_device)

    # Ảnh áo
    cloth_path = os.path.join(opt_config.dataroot, CLOTHES_DIR_IN_DATAROOT, cloth_basename)
    if not os.path.exists(cloth_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh áo '{cloth_basename}' tại '{cloth_path}'.")
    cloth_pil = Image.open(cloth_path).convert('RGB')
    C_tensor = get_img_transform()(cloth_pil).unsqueeze(0).to(current_device) 


    edge_basename = cloth_basename 
    edge_path = os.path.join(opt_config.dataroot, CLOTHES_EDGE_DIR_IN_DATAROOT, edge_basename)
    if not os.path.exists(edge_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh biên '{edge_basename}' tại '{edge_path}'.")
    edge_pil = Image.open(edge_path).convert('L') 
    E_tensor = get_edge_transform()(edge_pil).unsqueeze(0).to(current_device) 


    with torch.no_grad():
        # Tạo biên nhị phân từ E_tensor 
        edge_binary = (E_tensor > 0.5).float() 
        
        # Áp dụng biên nhị phân vào ảnh áo để warping
        clothes_for_warp = C_tensor * edge_binary 

        flow_out = warp_model_loaded(I_tensor, clothes_for_warp)
        warped_cloth, last_flow = flow_out 
        
        warped_edge = F.grid_sample(E_tensor, last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros', align_corners=False)

        gen_inputs = torch.cat([I_tensor, warped_cloth, warped_edge], 1)
        gen_outputs = gen_model_loaded(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1) 
        p_rendered = torch.tanh(p_rendered) 
        m_composite = torch.sigmoid(m_composite) 
        # Kết hợp mặt nạ tổng hợp với warped_edge
        m_composite_final = m_composite * warped_edge
        
        p_tryon = warped_cloth * m_composite_final + p_rendered * (1 - m_composite_final)

    safe_cloth_basename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in cloth_basename)
    output_filename = f"result_{request_id}_{safe_cloth_basename}"
    output_path = os.path.join(final_result_dir, output_filename)

    # Chuyển tensor p_tryon thành ảnh có thể lưu
    result_img_tensor = p_tryon[0].cpu().detach() 
    result_img_np = (result_img_tensor.permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0 
    result_img_np = result_img_np.astype(np.uint8)
    result_img_bgr = cv2.cvtColor(result_img_np, cv2.COLOR_RGB2BGR) 
    
    cv2.imwrite(output_path, result_img_bgr)

    return output_path


@app.post("/try-on/")
async def try_on_api_endpoint(
    person_image: UploadFile = File(..., description="Ảnh người mẫu tải lên (JPG, PNG)"),
    cloth_image_name: str = Form(..., description="Tên file của ảnh áo (ví dụ: '000001_1.jpg')")
):

    original_person_filename = person_image.filename if person_image.filename else "uploaded_person.jpg"
    temp_person_image_name = f"{uuid.uuid4()}_{original_person_filename}"
    temp_person_image_path = os.path.join(RAW_PERSON_UPLOAD_DIR, temp_person_image_name)
    
    try:
        with open(temp_person_image_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu ảnh người tải lên: {str(e)}")
    finally:
        person_image.file.close() 

    # Kiểm tra xem models đã được tải chưa 
    if not all([app_state.opt, app_state.warp_model, app_state.gen_model]):
        raise HTTPException(status_code=503, detail="Models chưa sẵn sàng. Vui lòng thử lại sau giây lát.")

    try:
        start_time = time.time()
        # Gọi hàm xử lý chính
        result_image_path = perform_viton_tryon_logic(
            opt_config=app_state.opt,
            raw_person_image_path=temp_person_image_path,
            cloth_basename=cloth_image_name,
            unet_output_dir=UNET_PERSON_OUTPUT_DIR,
            final_result_dir=FINAL_RESULTS_DIR,
            warp_model_loaded=app_state.warp_model,
            gen_model_loaded=app_state.gen_model,
            unet_model_instance=unet_model_global, 
            current_device=app_state.device
        )
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Quá trình thử đồ mất {processing_time:.2f} giây.")

        # Trả về file ảnh kết quả
        return FileResponse(
            result_image_path, 
            media_type="image/jpeg", 
            filename=os.path.basename(result_image_path), 
            content_disposition_type="inline" 
        )
    
    except HTTPException as http_exc: 
        raise http_exc
    except Exception as e:

        import traceback
        error_details = f"Lỗi trong quá trình thử đồ: {e}\n{traceback.format_exc()}"
        print(error_details)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ trong quá trình thử đồ: {str(e)}")
    finally:

        if os.path.exists(temp_person_image_path):
            try:
                os.remove(temp_person_image_path)
            except Exception as e_clean:
                print(f"Lỗi khi dọn dẹp file tạm {temp_person_image_path}: {e_clean}")

@app.get("/clothes/")
async def list_available_clothes_api():
    """
    Trả về danh sách các ảnh áo có sẵn trong thư mục dataset/test_clothes
    mà có ảnh biên tương ứng trong dataset/test_edge.
    """
    if not app_state.opt or not app_state.opt.dataroot:
         raise HTTPException(status_code=503, detail="Cấu hình chưa sẵn sàng.")

    clothes_actual_dir = os.path.join(app_state.opt.dataroot, CLOTHES_DIR_IN_DATAROOT)
    edge_actual_dir = os.path.join(app_state.opt.dataroot, CLOTHES_EDGE_DIR_IN_DATAROOT)

    if not os.path.isdir(clothes_actual_dir):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy thư mục ảnh áo: {clothes_actual_dir}")
    if not os.path.isdir(edge_actual_dir):
        print(f"Cảnh báo: Không tìm thấy thư mục ảnh biên: {edge_actual_dir}. Sẽ không có áo nào được liệt kê nếu yêu cầu biên.")

    try:
        image_extensions = ('.jpg', '.jpeg', '.png')
        cloth_files_on_disk = [
            f for f in os.listdir(clothes_actual_dir)
            if os.path.isfile(os.path.join(clothes_actual_dir, f)) and f.lower().endswith(image_extensions)
        ]
        
        valid_clothes_with_edges = []
        for cloth_file in cloth_files_on_disk:
            corresponding_edge_path = os.path.join(edge_actual_dir, cloth_file)
            if os.path.exists(corresponding_edge_path):
                valid_clothes_with_edges.append(cloth_file)
            else:
                print(f"Thông tin: Ảnh áo '{cloth_file}' có tồn tại, nhưng không tìm thấy ảnh biên tương ứng tại '{corresponding_edge_path}'. Bỏ qua.")
        
        return JSONResponse(content={"available_clothes": valid_clothes_with_edges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi liệt kê danh sách áo: {str(e)}")
