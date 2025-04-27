import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for an hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns - 1e5 * Y / ||Y|| ** 2 to ensure exp(<X, Y>) = 0
    Raises a ValueError if no such hyperplane is found
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(name, func):
    """
    Decorator to udpate the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k such that exp(<q, k>) = 0 to fake head-wise compression
    This solution is not optimal as it does not reduce peak memory and slightly increase runtime
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif module.masked_key_indices is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        if name == "flex_attention":
            """
            def flex_attention_forward(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Union[torch.Tensor, "BlockMask"],
                scaling: Optional[float] = None,
                softcap: Optional[float] = None,
                head_mask: Optional[torch.Tensor] = None,
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:

                returns torch.Size([1, 4, 7, 695])
            """
            scaling = kwargs.get("scaling")
            del kwargs["scaling"]
            kernel_options = {
                "BLOCK_M": 32,
                # "BLOCK_M": 64,
                "BLOCK_N": 32,
                # "BLOCK_N": 64,
                "BLOCK_M1": 2,
                "BLOCK_N1": 4,
                "BLOCK_M2": 4,
                "BLOCK_N2": 2,
            }
            kwargs["kernel_options"] = kernel_options
            result1, _ = func(module, query, key, value, attention_mask, scaling, None, None, **kwargs)
            return result1, None

        """
            def sdpa_attention_forward(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                dropout: float = 0.0,
                scaling: Optional[float] = None,
                is_causal: Optional[bool] = None,
                **kwargs,
            ) -> Tuple[torch.Tensor, None]:

            returns torch.Size([1, 28, 695])
        """
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Add the attention_patch decorator to functions in ALL_ATTENTION_FUNCTIONS
    """

    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        print(f"Patched {name}")
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(name, func)
