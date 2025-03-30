from torch.fx import Node
from typing import Sequence, Optional
from torch.utils._pytree import tree_flatten
import torch

def get_args_of_node_type(node: Node) -> Sequence[Node]:
    return [x for x in tree_flatten((node.args, node.kwargs))[0]
        if isinstance(x, Node)]

def use_tangent(node: Node) -> bool:
    """
    Whether the fx node uses tangent input.
    """

    return any(
        arg.op == "placeholder" and "tangent" in arg.target
        for arg in get_args_of_node_type(node)
    )

def compute_tensor_size(*args, count_bytes=True, **kwargs):
    """
    Compute total tensor sizes from fx.Node in args & kwargs.
    """
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node_arg(arg)) is None:
            continue
        tot += fake_tensor.numel() * (fake_tensor.dtype.itemsize if count_bytes else 1)
    return tot

def get_fake_tensor_from_node_arg(node: torch.fx.node.Argument) -> Optional[torch.Tensor]:
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)
        or not isinstance(node.meta["val"], torch.Tensor)
    ):
        return None
    return node.meta["val"]

def get_nodes_with_chunking_meta(graph: torch.fx.Graph) -> Sequence[Node]:
    from .core import get_chunking_meta

    output = []
    for node in graph.nodes:
        if get_chunking_meta(node):
            output.append(node)
    return output

def format_node_with_chunking_meta(node: torch.fx.Node, include_args=False):
    """
    Print the node with chunking metadata for the current node if exists.

    If include_args is True, also print chuning metadata for Node arguments.
    """
    from torch._inductor.runtime.runtime_utils import green_text
    from .core import get_chunking_meta
    fake_tensor = get_fake_tensor_from_node_arg(node)
    shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
    print(f"  {shape} {node.format_node()}")

    if meta := get_chunking_meta(node):
        print(f"    {green_text(str(meta))}")
   
    if include_args:
        for arg in get_args_of_node_type(node):
            if arg_meta := get_chunking_meta(arg):
                print(f"    {arg}: {green_text(str(arg_meta))}")

def has_any_chunking_meta(*node_list):
    from .core import get_chunking_meta
    return any(get_chunking_meta(node) for node in node_list)

def get_first_chunking_meta(*node_list):
    """
    Get the first non-none chunking metadata if there is any.
    """
    from .core import get_chunking_meta
    for node in node_list:
        if (meta := get_chunking_meta(node)) is not None:
            return meta

    return None

def get_scale_by_from_metas(*metas):
    """
    If there are multiple ChunkingMeta has the scale_by field,
    raise a CantChunk exception.

    If no ChunkingMeta has scale_by field, return None.
    Other wise return the only scale_by field.
    """
    from .core import CantChunk
    scale_by_list = []

    # don't do dedup on the scale_by field on purpose for this API
    for meta in metas:
        if meta.scale_by is not None:
            scale_by_list.append(meta.scale_by)

    if len(scale_by_list) > 1:
        raise CantChunk("Multiple scale_by")

    return scale_by_list[0] if len(scale_by_list) == 1 else None

def get_scale_by_from_node(node):
    from .core import get_chunking_meta
    meta = get_chunking_meta(node)
    return meta.scale_by if meta is not None else None

def get_node_is_scalar(nodes):
    """
    Returns a dict map a node to 'is_scalar'.
    """
    node_is_scalar = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        node_is_scalar[node] = ft.numel() == 1
    return node_is_scalar

def is_chunked_by_dim(node, dim):
    from .core import get_chunking_meta
    meta = get_chunking_meta(node)
    return meta and meta.chunk_by_dim(dim)
