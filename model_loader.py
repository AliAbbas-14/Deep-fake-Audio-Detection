import joblib
import torch
from huggingface_hub import hf_hub_download

def load_sklearn_model(repo, filename):
    model_path = hf_hub_download(repo, filename)
    return joblib.load(model_path)

def load_torch_model(repo, filename, model_class):
    model_path = hf_hub_download(repo, filename)
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
