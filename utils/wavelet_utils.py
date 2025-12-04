"""Utilities for Wavelet transformations (Haar) and manipulations."""

import torch
import numpy as np
import math

def generate_haar_matrix(T: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates the single-level Haar Wavelet Transform matrix.

    Args:
        T (int): Number of time steps (must be even).
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: The Haar Wavelet Transform matrix of shape [T, T].
                      This tensor does not require gradients.
    """
    if T % 2 != 0:
        raise ValueError(f"Time dimension T ({T}) must be even for this Haar matrix implementation.")

    # Matrix W such that C = W @ P
    # Rows 0:T/2-1 are for approximation coeffs (low-pass)
    # Rows T/2:T-1 are for detail coeffs (high-pass)
    matrix = np.zeros((T, T))
    sqrt2_inv = 1.0 / math.sqrt(2.0) # Precompute for efficiency

    for i in range(T // 2):
        # Approximation coefficient row i
        matrix[i, 2*i]     = sqrt2_inv
        matrix[i, 2*i + 1] = sqrt2_inv

        # Detail coefficient row i (index i + T // 2)
        matrix[i + T // 2, 2*i]     = sqrt2_inv
        matrix[i + T // 2, 2*i + 1] = -sqrt2_inv

    # Convert to torch tensor
    haar_matrix = torch.from_numpy(matrix).float().to(device)

    # Ensure it's treated as a constant
    haar_matrix.requires_grad_(False)

    return haar_matrix

def points_to_wavelet_coeffs(points: torch.Tensor, wavelet_matrix: torch.Tensor) -> torch.Tensor:
    """
    Transforms points to wavelet coefficients using the provided transform matrix.

    Args:
        points (torch.Tensor): The original points tensor of shape [B, T, 3].
        wavelet_matrix (torch.Tensor): The wavelet transform matrix (e.g., Haar) of shape [T, T].

    Returns:
        torch.Tensor: The calculated wavelet coefficients tensor of shape [B, T, 3].
    """
    B, T, _ = points.shape
    T_w, T_w_ = wavelet_matrix.shape
    if T != T_w or T != T_w_:
        raise ValueError(f"Mismatch in time dimension: points have {T} steps, matrix is {T_w}x{T_w_}")

    # Transform: C[b,t_c,d] = sum_t( W[t_c, t] * P[b,t,d] )
    # Use einsum: [T, T] x [B, T, 3] -> [B, T, 3]
    # Note: PyTorch matmul handles batching correctly: [T, T] @ [B, T, 3]^T -> [T, B, 3] -> permute
    coeffs = torch.matmul(points.permute(0, 2, 1), wavelet_matrix.T).permute(0, 2, 1)
    # coeffs = torch.einsum('ct,btx->bcx', wavelet_matrix, points) # Equivalent using einsum

    # Note: Shape remains [B, T, 3], where the first T/2 are approx coeffs, last T/2 are detail coeffs.

    return coeffs

def mask_wavelet_coeffs(coeffs: torch.Tensor, mask: torch.Tensor, training: bool) -> tuple[torch.Tensor, int | None]:
    """
    Applies a mask to the wavelet coefficients.

    Args:
        coeffs (torch.Tensor): Wavelet coefficients of shape [B, T, 3].
        mask (torch.Tensor): Boolean or float mask of shape [B, T].
        training (bool): Flag indicating if in training or testing mode.

    Returns:
        tuple[torch.Tensor, int | None]:
            If training=True: Returns the masked coefficients (shape [B, T, 3]) where
                              masked elements are zeroed, and None for K.
            If training=False: Returns a flattened tensor of non-masked coefficients
                               (shape [K, 3], where K is the number of True elements
                               in the mask) and the count K.
    """
    B, T, _ = coeffs.shape
    if mask.shape != (B, T):
        raise ValueError(f"Mask shape {mask.shape} mismatch with coeffs shape {coeffs.shape[:2]}")

    mask_bool = mask.bool() # Ensure boolean mask for indexing

    if training:
        # Apply mask by multiplication (zeros out masked elements)
        # Ensure mask is broadcastable [B, T, 1]
        masked_coeffs = coeffs * mask_bool.unsqueeze(-1).float()
        return masked_coeffs
    else:
        # Testing: Select only non-masked elements and flatten
        compact_coeffs = coeffs[mask_bool] # Select based on boolean mask directly [K, 3]
        return compact_coeffs

def restore_masked_wavelet_coeffs(compact_coeffs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Restores the original [B, T, 3] shape from compacted coefficients and a mask.
    Fills masked positions with zeros.

    Args:
        compact_coeffs (torch.Tensor): Compacted coefficients of shape [K, 3].
        mask (torch.Tensor): The original boolean mask of shape [B, T] used
                             for compaction.

    Returns:
        torch.Tensor: The restored coefficients tensor of shape [B, T, 3] with
                      zeros at masked locations.
    """
    B, T = mask.shape
    if mask.dtype != torch.bool:
         mask = mask.bool() # Ensure boolean mask

    # Create a zero tensor with the target shape
    restored_coeffs = torch.zeros(B, T, 3, device=compact_coeffs.device, dtype=compact_coeffs.dtype)

    # Place the compact coefficients into the zero tensor at unmasked locations
    # Ensure the number of True elements in mask matches K
    K = mask.sum().item()
    if K != compact_coeffs.shape[0]:
        raise ValueError(f"Number of True elements in mask ({K}) does not match "
                         f"the first dimension of compact_coeffs ({compact_coeffs.shape[0]})")

    restored_coeffs[mask] = compact_coeffs

    return restored_coeffs

def wavelet_coeffs_to_points(coeffs: torch.Tensor, wavelet_matrix: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the original points from wavelet coefficients using the inverse transform.
    For orthogonal transforms like Haar, inverse(W) = W.T.
    Computes P = W.T @ C (per batch element).

    Args:
        coeffs (torch.Tensor): Wavelet coefficients tensor of shape [B, T, 3].
        wavelet_matrix (torch.Tensor): The Haar wavelet transform matrix of shape [T, T].

    Returns:
        torch.Tensor: The reconstructed points tensor of shape [B, T, 3].
    """
    B, T_c, _ = coeffs.shape
    T_w, T_w_ = wavelet_matrix.shape
    if T_c != T_w or T_c != T_w_:
        raise ValueError(f"Mismatch in dimension: coeffs have {T_c}, matrix is {T_w}x{T_w_}")

    # Inverse transform: P[b,t,d] = sum_tc( W.T[t, tc] * C[b,tc,d] )
    # Use einsum: [T, T] x [B, T, 3] -> [B, T, 3] (where first T is from W.T)
    wavelet_matrix_inv = wavelet_matrix.T # Since Haar matrix is orthogonal
    # points = torch.einsum('tc,bcx->btx', wavelet_matrix_inv, coeffs)
    # Use matmul: (P^T = C^T @ W) -> P = (C^T @ W)^T
    points = torch.matmul(coeffs.permute(0, 2, 1), wavelet_matrix).permute(0, 2, 1) # Correct: [B, 3, T] @ [T, T] -> [B, 3, T] -> [B, T, 3]


    return points
