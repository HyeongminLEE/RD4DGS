"""Utilities for B-spline transformations and manipulations."""

import torch
import numpy as np
from scipy.interpolate import BSpline

def generate_bspline_matrix(T: int, M: int, degree: int = 3, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generates the B-spline basis matrix.

    Args:
        T (int): Number of time steps (output dimension).
        M (int): Number of control points (input dimension).
        degree (int): Degree of the B-spline. Defaults to 3 (cubic).
        device (torch.device): The device to place the output tensor on.

    Returns:
        torch.Tensor: The B-spline basis matrix of shape [T, M],
                      representing the basis functions evaluated at T time points.
                      This tensor does not require gradients.
    """
    # Ensure M is sufficient for the degree
    if M <= degree:
        raise ValueError(f"Number of control points M ({M}) must be greater than degree ({degree})")

    # Generate knot vector (open uniform)
    # Knots range from 0 to M - degree
    # Total knots = M + degree + 1
    knots = np.concatenate(([0] * degree, np.arange(M - degree + 1), [M - degree] * degree))

    # Time points where basis functions are evaluated (normalized 0 to M - degree)
    t_points = np.linspace(0, M - degree, T)

    # Create the B-spline basis functions using scipy
    # The i-th basis function corresponds to the i-th control point
    basis_matrix_np = np.zeros((T, M))
    for i in range(M):
        # Construct the i-th basis function
        # Use identity matrix for coefficients c: B(t) = sum(c_i * N_{i, p}(t))
        # To get N_{i, p}(t), set c_i = 1 and others = 0
        c = np.zeros(M)
        c[i] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=False)
        basis_matrix_np[:, i] = spl(t_points)

    # Handle potential NaNs at boundaries introduced by extrapolation=False, replace with 0
    basis_matrix_np = np.nan_to_num(basis_matrix_np)

    # Convert to torch tensor and move to device
    bspline_matrix = torch.from_numpy(basis_matrix_np).float().to(device)

    # Ensure it's treated as a constant (no gradient tracking)
    bspline_matrix.requires_grad_(False)

    return bspline_matrix

def points_to_bspline_params(points: torch.Tensor, bspline_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates B-spline control point parameters from original points
    using least squares. Solves B @ C = P for C, where B is the B-spline matrix,
    P are the points, and C are the parameters.

    Args:
        points (torch.Tensor): The original points tensor of shape [B, T, 3].
        bspline_matrix (torch.Tensor): The B-spline basis matrix of shape [T, M].

    Returns:
        torch.Tensor: The calculated B-spline parameters tensor of shape [B, M, 3].
    """
    B, T, _ = points.shape
    T_b, M = bspline_matrix.shape
    assert T == T_b, "Time dimension mismatch."

    # Compute pseudo-inverse of basis_matrix => shape [M, T].
    BT = bspline_matrix.transpose(0, 1)  # [M, T]
    invBTB = torch.inverse(BT @ bspline_matrix)  # [M, M]
    basis_pinv = invBTB @ BT  # [M, T]

    # Encode each sequence in the batch.
    control_points_list = []
    for b_idx in range(B):
        seq_b = points[b_idx]  # [T, 3]
        ctrl_b = basis_pinv @ seq_b  # [M, 3]
        control_points_list.append(ctrl_b)

    # Stack into [B, M, 3]
    control_points = torch.stack(control_points_list, dim=0)
    return control_points

def mask_bspline_params(params: torch.Tensor, mask: torch.Tensor, training: bool) -> tuple[torch.Tensor, int | None]:
    """
    Applies a mask to the B-spline parameters.

    Args:
        params (torch.Tensor): B-spline parameters of shape [B, M, 3].
        mask (torch.Tensor): Boolean or float mask of shape [B, M].
        training (bool): Flag indicating if in training or testing mode.

    Returns:
        tuple[torch.Tensor, int | None]:
            If training=True: Returns the masked parameters (shape [B, M, 3]) where
                              masked elements are zeroed, and None for K.
            If training=False: Returns a flattened tensor of non-masked parameters
                               (shape [K, 3], where K is the number of True elements
                               in the mask) and the count K.
    """
    B, M, _ = params.shape
    if mask.shape != (B, M):
        raise ValueError(f"Mask shape {mask.shape} mismatch with params shape {params.shape[:2]}")

    mask_bool = mask.bool() # Ensure boolean mask for indexing

    if training:
        # Apply mask by multiplication (zeros out masked elements)
        # Ensure mask is broadcastable [B, M, 1]
        masked_params = params * mask_bool.unsqueeze(-1).float()
        return masked_params
    else:
        # Testing: Select only non-masked elements and flatten
        compact_params = params[mask_bool] # Select based on boolean mask directly [K, 3]
        return compact_params

def restore_masked_bspline_params(compact_params: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Restores the original [B, M, 3] shape from compacted parameters and a mask.
    Fills masked positions with zeros.

    Args:
        compact_params (torch.Tensor): Compacted parameters of shape [K, 3].
        mask (torch.Tensor): The original boolean mask of shape [B, M] used
                             for compaction.

    Returns:
        torch.Tensor: The restored parameters tensor of shape [B, M, 3] with
                      zeros at masked locations.
    """
    B, M = mask.shape
    if mask.dtype != torch.bool:
         mask = mask.bool() # Ensure boolean mask

    # Create a zero tensor with the target shape
    restored_params = torch.zeros(B, M, 3, device=compact_params.device, dtype=compact_params.dtype)

    # Place the compact parameters into the zero tensor at unmasked locations
    # Ensure the number of True elements in mask matches K
    K = mask.sum().item()
    if K != compact_params.shape[0]:
        raise ValueError(f"Number of True elements in mask ({K}) does not match "
                         f"the first dimension of compact_params ({compact_params.shape[0]})")

    restored_params[mask] = compact_params

    return restored_params

def bspline_params_to_points(params: torch.Tensor, bspline_matrix: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the original points from B-spline parameters and the basis matrix.
    Computes P = B @ C (per batch element).

    Args:
        params (torch.Tensor): B-spline parameters tensor of shape [B, M, 3].
        bspline_matrix (torch.Tensor): The B-spline basis matrix of shape [T, M].

    Returns:
        torch.Tensor: The reconstructed points tensor of shape [B, T, 3].
    """
    B, M, _ = params.shape
    T, _M = bspline_matrix.shape
    if _M != M:
        raise ValueError(f"Mismatch in control point dimension: params have {M} points, matrix expects {_M}")

    # Reconstruct points: P[b,t,d] = sum_m( B[t,m] * C[b,m,d] )
    # Use einsum for clear and efficient batch matrix multiplication: [T, M] x [B, M, 3] -> [B, T, 3]
    points = torch.einsum('tm,bmx->btx', bspline_matrix, params)
    # Alternative using matmul: requires permutations
    # P = (B @ C^T)^T  where C is [B, M, 3] -> permute to [B, 3, M]
    # params_permuted = params.permute(0, 2, 1) # [B, 3, M]
    # result_permuted = torch.matmul(bspline_matrix, params_permuted) # Incorrect: [T, M] @ [B, 3, M]
    # Correct matmul: Loop or batch matmul if possible
    # points = torch.matmul(bspline_matrix, params.permute(2, 0, 1)).permute(1, 2, 0)? No.
    # points_list = []
    # for b in range(B):
    #     points_list.append(torch.matmul(bspline_matrix, params[b]).unsqueeze(0)) # [T, M] @ [M, 3] -> [T, 3] -> [1, T, 3]
    # points = torch.cat(points_list, dim=0)

    return points
