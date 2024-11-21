from transformers import AutoModelForMaskedLM
import torch
import copy 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {device}")

paths = [
#Replace with your model weights
    r"C:\Users\Sri Seshadri\Desktop\Sri RLXF\trained_models\SFT_ESM2_35M\sft_updated_esm2_version_0.pt",
    r"C:\Users\Sri Seshadri\Desktop\Sri RLXF\trained_models\SFT_ESM2_35M\sft_updated_esm2_version_1.pt",
    r"C:\Users\Sri Seshadri\Desktop\Sri RLXF\trained_models\SFT_ESM2_35M\sft_updated_esm2_version_2.pt"
]

def load_and_average_checkpoints(checkpoint_paths, device='cuda'):
    models = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device)
        base_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
        base_model.load_state_dict(checkpoint, strict=False)
        models.append(base_model)
        
        print(f"\nWeights from {path}:")
        print(checkpoint['esm.embeddings.word_embeddings.weight'][:5, :5])
    
    soup_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for name, param in soup_model.named_parameters():
            param_sum = torch.zeros_like(param, device=device)
            for model in models:
                param_sum += model.state_dict()[name]
            param.copy_(param_sum / len(models))
    
    print("\nAveraged weights:")
    print(soup_model.state_dict()['esm.embeddings.word_embeddings.weight'][:5, :5])
    
    return soup_model


soup_model = load_and_average_checkpoints(paths, save_path="esm2_35M_souped.pt")
