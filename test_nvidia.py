import torch

if torch.cuda.is_available():
    print("Nvidia GPU is available and being used.")
else:
    print("No Nvidia GPU available.")