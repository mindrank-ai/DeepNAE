import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def apply_activation_to_fingerprint(fingerprint, activation_function='relu'):
    """Apply an activation function to the Morgan fingerprint."""
    # Convert the fingerprint to a PyTorch tensor
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)
    
    # Apply the chosen activation function
    if activation_function == 'relu':
        activated_fingerprint = F.relu(fingerprint_tensor)
    elif activation_function == 'sigmoid':
        activated_fingerprint = torch.sigmoid(fingerprint_tensor)
    elif activation_function == 'tanh':
        activated_fingerprint = torch.tanh(fingerprint_tensor)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")
    
    return activated_fingerprint


def molecular_attention(FP_list1, FP_list2, cuda_index, apply_activation=False, activation_function='relu'):
    """
    Calculates the molecular attention-like mechanism using Tanimoto similarity.
    This computes the similarity between two sets of chemical fingerprints (Query and Key)
    and returns a normalized similarity matrix, with an optional activation function.

    Args:
        FP_list1 (list): List of query fingerprints.
        FP_list2 (list): List of key fingerprints.
        cuda_index (int): The index of the GPU device for computation.
        apply_activation (bool): Whether to apply an activation function to the fingerprints.
        activation_function (str): The activation function to apply, can be 'relu', 'sigmoid', or 'tanh'.

    Returns:
        np.ndarray: Normalized similarity matrix where each element represents the 
                    Tanimoto similarity between a fingerprint from FP_list1 and FP_list2.
    """
    # Initialize a similarity matrix with zeros
    sim_matrix = np.zeros([len(FP_list1), len(FP_list2)])

    # Convert the fingerprint lists to PyTorch tensors and move them to the specified GPU
    FP_tn_list = torch.ByteTensor(FP_list1).to(cuda_index)
    FP_list2 = torch.ByteTensor(FP_list2).to(cuda_index)

    # Optionally apply activation function
    if apply_activation:
        FP_tn_list = [apply_activation_to_fingerprint(fps, activation_function) for fps in FP_tn_list]
        FP_list2 = [apply_activation_to_fingerprint(fps, activation_function) for fps in FP_list2]
    
    # Iterate over each fingerprint in FP_list1 to calculate the similarity
    for i in tqdm(range(len(FP_tn_list))):
        ref_fps = FP_tn_list[i]
        
        # Compute Tanimoto similarity between the query (ref_fps) and all keys (FP_list2)
        intersection = torch.sum(ref_fps.unsqueeze(0) & FP_list2, dim=-1, dtype=torch.float16)
        union = torch.sum(ref_fps.unsqueeze(0) | FP_list2, dim=-1, dtype=torch.float16)
        
        # Calculate Tanimoto similarity (avoid division by zero)
        smi_map = intersection / (union)
        sim_matrix[i] = smi_map.cpu().numpy()

    # Clear CUDA memory to prevent memory overflow
    if cuda_index != 'cpu':
        with torch.cuda.device(cuda_index):
            torch.cuda.empty_cache()

    return sim_matrix