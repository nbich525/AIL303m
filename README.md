# AIL303m

# 🖥️ Installation

## 1. Create a Conda Environment
```bash
conda create -n tryon python=3.6
conda activate tryon
```

## 2. Install PyTorch
```bash
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
```

## 3. Install Other Dependencies
```bash
conda install cupy
pip install opencv-python
```

## 4. Clone the Repository
```bash
git clone https://github.com/nbich525/AIL303m.git
cd AIL303m
```

## 🔍 Testing

1. **Download the Checkpoints**  
  Download the checkpoints from [here](https://drive.google.com/drive/folders/15AbTw16w13dN1hY430flBZbe1EfumJT7) and create a new folder `checkpoints`.  
  The folder `checkpoints` should contain `warp_model_final.pth` and `gen_model_final.pth`.

2. **Run the Test**  
   Run the following command to test the saved model:
```bash
python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0
```

