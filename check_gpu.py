import torch
try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    print("Attempting small tensor operation on GPU...")
    x = torch.tensor([1.0, 2.0]).cuda()
    y = x * 2
    print(f"Success! Result: {y}")
except Exception as e:
    print(f"GPU FAILED: {e}")