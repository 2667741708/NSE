import torch
try:
    import faiss
    print("FAISS imported")
    print("FAISS GPU:", hasattr(faiss, "StandardGpuResources"))
except Exception as e:
    print("FAISS error:", e)

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
