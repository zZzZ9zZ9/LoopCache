import torch


def kv_bytes_per_token(config, depth, dtype):
    if type(depth) is not int or depth < 1:
        raise ValueError("Recurrence depth must be a positive integer")
    if config.model_type == "ouro":
        slots = depth * config.num_hidden_layers
    elif config.model_type == "huginn_raven":
        slots = (
            config.n_layers_in_prelude
            + depth * config.n_layers_in_recurrent_block
            + config.n_layers_in_coda
        )
    else:
        raise ValueError(f"Unknown model type {config.model_type!r}")
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return 2 * slots * config.num_key_value_heads * head_dim * dtype.itemsize


def available_bytes(device):
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    cached = max(0, reserved - allocated)
    get_fraction = getattr(torch.cuda.memory, "get_per_process_memory_fraction", None)
    if get_fraction is not None:
        headroom = max(0, int(total * get_fraction(device)) - reserved)
        free = min(free, headroom)
    return free + cached
