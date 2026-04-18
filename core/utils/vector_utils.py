def ensure_list_vector(vec):
    """
    Đảm bảo vector luôn là list[float]
    """
    if vec is None:
        return None

    # numpy → list
    if hasattr(vec, "tolist"):
        return vec.tolist()

    # torch tensor → numpy → list
    if hasattr(vec, "cpu"):
        return vec.cpu().numpy().tolist()

    # đã là list
    if isinstance(vec, list):
        return vec

    # fallback
    return list(vec)