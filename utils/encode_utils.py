from argparse import ArgumentParser
import os
import struct
import numpy as np
import torch
from utils.entropy_model import DiscreteUnconditionalEntropyModel, Softmax

def remove_negatives(codebook):
    mask = codebook < 0
    codebook[mask] = 1e-8

def pack_string(string):
    bit_stream = struct.pack(f'>I', len(string))
    bit_stream += struct.pack(f'>{len(string)}s', string)
    return bit_stream

def unpack_string(bit_stream):
    s1 = struct.calcsize('I')
    s2 = struct.calcsize('s')
    length = struct.unpack(f'>I', bit_stream[:s1])[0]
    string = struct.unpack(f'>{length}s', bit_stream[s1:s1+s2*length])[0]
    return string, bit_stream[s1+s2*length:]

def pack_strings(strings):
    bit_stream = b''
    for string in strings:
        bit_stream += pack_string(string)
    return bit_stream

def unpack_strings(bit_stream, n):
    strings = []
    for i in range(n):
        string, bit_stream = unpack_string(bit_stream)
        strings.append(string)
    return strings, bit_stream

def pack_uints(uints):
    bit_stream = struct.pack(f'>{len(uints)}I', *uints)
    return bit_stream

def unpack_uints(bit_stream, n):
    s1 = struct.calcsize('I')
    uints = struct.unpack(f'>{n}I', bit_stream[:n*s1])
    return uints, bit_stream[n*s1:]

def pack_floats(floats):
    bit_stream = struct.pack(f'>{len(floats)}f', *floats)
    return bit_stream

def unpack_floats(bit_stream, n):
    s1 = struct.calcsize('f')
    floats = struct.unpack(f'>{n}f', bit_stream[:n*s1])
    return floats, bit_stream[n*s1:]


def shmask_sort(sh_bitmask:dict):
    shmask_sorted = np.zeros_like(sh_bitmask['sh1'], dtype=int)

    for key in sh_bitmask:
        shmask_sorted = (shmask_sorted << 1) | sh_bitmask[key].astype(int)

    sort_idx = np.argsort(shmask_sorted)
    num_gaussians = shmask_sorted.shape[0]
    shmask_sorted = shmask_sorted[sort_idx]
    boundaries = np.where(np.diff(shmask_sorted))[0] + 1
    boundaries = np.append(boundaries, num_gaussians)

    shmask_sorted = np.insert(shmask_sorted, 0, 0)
    shmask_sorted = np.append(shmask_sorted, 7)
    diff_array = np.diff(shmask_sorted)
    indices = np.flatnonzero(diff_array)
    boundaries = []

    for i in indices:
        value = diff_array[i]
        boundaries.extend([i] * value)

    boundaries = np.array(boundaries)
    if boundaries[-1] != shmask_sorted.shape[0]:
        boundaries = np.append(boundaries, num_gaussians)

    return sort_idx, boundaries

def entropy_coding(attr_index, attr_logits, range_coder_precision=16):
    em = DiscreteUnconditionalEntropyModel(Softmax(attr_logits), range_coder_precision)
    em.get_ready_for_compression()
    bits_str = em.compress(attr_index)
    decode_results = em.decompress(bits_str, attr_index.shape)
    return bits_str

def entropy_decoding(bits_str, attr_index_shape, attr_logits, range_coder_precision=16):
    em = DiscreteUnconditionalEntropyModel(Softmax(attr_logits), range_coder_precision)
    em.get_ready_for_compression()
    decode_results = em.decompress(bits_str, attr_index_shape)
    return decode_results

def shrink_codebook(codebook_dict: dict, logits_dict: dict, index_dict: dict):
    for vq_key in codebook_dict.keys():
        codebook = codebook_dict[vq_key]
        logits = logits_dict[vq_key]
        index_vector = index_dict[vq_key].squeeze(axis=-1)

        used_indices = np.unique(index_vector)
        print(vq_key, 'num of used codewords', len(used_indices))

        new_codebook = codebook[:, used_indices]
        new_logits = logits[:, used_indices]
        new_index_vector = np.expand_dims(np.searchsorted(used_indices, index_vector), 1)

        codebook_dict[vq_key] = new_codebook
        logits_dict[vq_key] = new_logits
        index_dict[vq_key] = new_index_vector

    return codebook_dict, logits_dict, index_dict

def opacity_quant(opacities, precision=2**8):
    opacities = np.nan_to_num(opacities, nan=0.0)
    min_opacity = np.min(opacities)
    max_opacity = np.max(opacities)
    step_size = (max_opacity - min_opacity) / (precision - 1)

    quantized_opacities = np.round((opacities - min_opacity) / step_size).astype(np.uint8)
    dequantized_opacities = quantized_opacities * step_size + min_opacity

    hist, _ = np.histogram(quantized_opacities, bins=precision, density=True)
    log_prob = np.expand_dims(np.log(hist + 1e-10), 0)

    return quantized_opacities, dequantized_opacities, log_prob, step_size, min_opacity

def opacity_dequant(quantized_opacities, min_opacity, step_size):
    dequantized_opacities = quantized_opacities * step_size + min_opacity
    return dequantized_opacities

def decode_header(header_path):
    """Decodes the header file containing boundaries and opacity parameters."""
    with open(header_path, 'rb') as f:
        header_bitstream = f.read()
    # Unpack boundaries (either 1 for old format or 8 for new)
    # Assuming the number of boundary values indicates the format
    # Check the expected size based on the number of floats (8 for new format)
    float_size = struct.calcsize('f')
    num_floats_expected = 6 # step_size and min_opacity for static, motion, center
    uint_section_size = len(header_bitstream) - num_floats_expected * float_size

    if uint_section_size == 8 * struct.calcsize('I'): # New format with 8 boundaries
        boundaries, header_bitstream = unpack_uints(header_bitstream, 8)
    elif uint_section_size == 1 * struct.calcsize('I'): # Potentially old format, handle appropriately or raise error
         boundaries, header_bitstream = unpack_uints(header_bitstream, 1)
         # If supporting old format, might need default values for other opacities
         print("Warning: Reading header potentially in old format.")
    else:
         raise ValueError("Invalid header size.")

    # Unpack step_size and min_opacity for all four opacity types
    params = {}
    keys = ['static', 'motion', 'center']
    for key in keys:
        step_size, header_bitstream = unpack_floats(header_bitstream, 1)
        min_opacity, header_bitstream = unpack_floats(header_bitstream, 1)
        params[f'step_size_{key}'] = step_size[0]
        params[f'min_opacity_{key}'] = min_opacity[0]

    boundaries = np.array(boundaries, dtype=np.int32)

    return boundaries, params

def get_index_shapes(boundaries, static_num, dynamic_num, keyframe_num):
    """Calculates the shapes for VQ and opacity indexes."""
    dec_num_gaussians = boundaries[-1]
    if dec_num_gaussians != (static_num + dynamic_num):
         # This might happen if pruning occurred before saving VQ info but static/dynamic counts weren't updated
         # Or if the boundaries/counts are read incorrectly.
         # Let's assume boundaries[-1] is the correct total number after pruning.
         print(f"Warning: Total gaussians from boundaries ({dec_num_gaussians}) doesn't match static+dynamic ({static_num + dynamic_num}). Using boundaries value.")
         # We might need a way to infer static/dynamic split post-pruning if not saved explicitly.
         # For now, assume static_num and dynamic_num passed are correct *after* pruning.
         if boundaries[-1] != static_num + dynamic_num:
              raise ValueError("Mismatch in gaussian counts after pruning.")


    shapes = {
        'scale': (dec_num_gaussians, ),
        'rot': (static_num + keyframe_num * dynamic_num, ), # Combined static and dynamic rotations before reshaping in decode
        'dc': (dec_num_gaussians, ),
        'opa': (static_num, ), # Static opacity only
        'opa_motion': (dynamic_num, ),
        'opa_center': (dynamic_num * 2, ),
        'opa_var': (dynamic_num * 2, )
    }

    if len(boundaries) == 1: # Handle case without SH sorting/masking
        shapes.update({
            'sh1': (dec_num_gaussians, ),
            'sh2': (dec_num_gaussians, ),
            'sh3': (dec_num_gaussians, ),
        })
    else: # Calculate SH shapes based on boundaries
        dec_sh1_len = boundaries[7] - boundaries[3]
        dec_sh2_len = boundaries[7] - boundaries[5] + boundaries[3] - boundaries[1]
        dec_sh3_len = boundaries[7] - boundaries[6] + boundaries[5] - boundaries[4] \
            + boundaries[3] - boundaries[2] + boundaries[1] - boundaries[0]
        shapes.update({
            'sh1': (dec_sh1_len, ),
            'sh2': (dec_sh2_len, ),
            'sh3': (dec_sh3_len, ),
        })

    return shapes


def decode_indexes(index_path, logits_dict, shape_dict):
    """Decodes the entropy-coded indexes from the bitstream."""
    # Include all keys that were encoded
    attr_keys = ['scale', 'rot', 'dc', 'sh1', 'sh2', 'sh3', 'opa', 'opa_motion', 'opa_center', 'opa_var']
    with open(index_path, "rb") as f:
        index_bitstream = f.read()
    dec_indexes = {}
    for i, attr_key in enumerate(attr_keys):
        if attr_key not in shape_dict:
             print(f"Warning: Shape for key '{attr_key}' not found in shape_dict. Skipping decoding.")
             continue
        if attr_key not in logits_dict:
             print(f"Warning: Logits for key '{attr_key}' not found in logits_dict. Skipping decoding.")
             continue

        dec_indexes_string, index_bitstream = unpack_string(index_bitstream)
        # Ensure the shape is correctly passed (it should be a tuple)
        current_shape = shape_dict[attr_key]
        if not isinstance(current_shape, tuple):
             # If shape is just a number, make it a tuple (e.g., (N,))
             current_shape = (current_shape,)

        dec_indexes[attr_key] = entropy_decoding(dec_indexes_string, current_shape, torch.Tensor(logits_dict[attr_key]))

    return dec_indexes

def get_sh_bitmask(boundaries):
    num_gaussians = boundaries[-1]
    if len(boundaries) == 1:
        return {sh_key: np.ones((num_gaussians, ), dtype=np.bool_) for sh_key in ['sh1', 'sh2', 'sh3']}
    shmask_sorted = np.zeros(num_gaussians, dtype=int)
    for i in range(7, -1, -1):
        shmask_sorted[:boundaries[i]] = i

    sh_bitmask = {sh_key: np.zeros(num_gaussians, dtype=bool) for sh_key in ['sh1', 'sh2', 'sh3']}

    sh_bitmask['sh1'] |= (shmask_sorted & (1 << 2)).astype(bool)
    sh_bitmask['sh2'] |= (shmask_sorted & (1 << 1)).astype(bool)
    sh_bitmask['sh3'] |= (shmask_sorted & (1 << 0)).astype(bool)

    return sh_bitmask
