python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0
# run in http://localhost:8000/docs
uvicorn api:app --reload --host 0.0.0.0 --port 8000