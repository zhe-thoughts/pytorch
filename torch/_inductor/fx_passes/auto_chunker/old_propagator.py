import dataclasses
from torch.fx import Node
import torch
from typing import Optional
from torch.utils._ordered_set import OrderedSet
import functools

from .collector import get_args_of_node_type, CantChunk, eligible_source_node_op_to_idx, get_fake_tensor_from_node_arg, compute_tensor_size
from .chunking_subgraph import ChunkingSubgraph
from .core import ChunkingMeta, get_chunking_meta, set_chunking_meta, copy_chunking_meta
from .utils import get_scale_by_from_metas

aten = torch.ops.aten
prims = torch.ops.prims

# Rules to propagate chunking metadata from inputs to the current node
propagate_rules = {
}

def _register_propagate_rule(aten_op, handler):
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]

    for op in aten_op:
        propagate_rules[op] = handler
    return handler

def register_propagate_rule(aten_op):
    return functools.partial(_register_propagate_rule, aten_op)


class Propagator:

    @classmethod
    def chunk_external_nodes(cls, chunking_subgraph, graph, chunking_subgraph_nodes: OrderedSet[Node]):
        """
        Find all nodes that are suppose to be input to the chunking
        subgraph. Add chunking metadata to them..
        """

        source_user = chunking_subgraph.source_user
        batch_size = source_user.meta["val"].size(0)
        assert source_user.target in eligible_source_node_op_to_idx

        # For source_user, only its input specificed by
        # eligible_source_node_op_to_idx[target] need to be chunked
        no_chunk_nodes = OrderedSet()
        for idx, node in enumerate(get_args_of_node_type(source_user)):
            if idx != eligible_source_node_op_to_idx[source_user.target]:
                no_chunk_nodes.add(node)

        external_args = OrderedSet()
        for node in chunking_subgraph_nodes:
            for idx, arg in enumerate(get_args_of_node_type(node)):
                if arg not in chunking_subgraph_nodes:
                    external_args.add(arg)

        with graph.inserting_before(source_user):
            for node in external_args:
                def _should_chunk(node):
                    if node.meta["val"].numel() == 1:
                        return False
                    
                    if node in no_chunk_nodes:
                        return False

                    # unwrap an permute
                    if node.target == aten.permute.default:
                        node = node.args[0]

                    if node in no_chunk_nodes:
                        return False

                    # TODO: we should have a general way to decide if we should chunk an external argument of not
                    # Don't chunk if current is 1D tensor and is used to add the source_user. This is the pattern
                    # for bias
                    if node.meta["val"].ndim == 1:
                        for user in node.users:
                            if user.target == aten.add.Tensor and user.args[0] == source_user and user.args[1] == node:
                                return False

                    # Sanity check the tensor size
                    if node.meta["val"].size(0) != batch_size:
                        raise CantChunk("First dimension does not match batch_size. {batch_size=} v.s. {node.meta['val'].size(0)}.")
                    return True

                chunking_subgraph.add_external_node(node)
                if _should_chunk(node):
                    # attach the chunking metadata
                    set_chunking_meta(node, chunk_dim=0)

    @classmethod
    def add_chunking_meta(cls, chunking_subgraph):
        graph = chunking_subgraph.parent_graph
        chunking_subgraph_nodes = chunking_subgraph.subgraph_nodes

        cls.chunk_external_nodes(chunking_subgraph, graph, chunking_subgraph_nodes)

        for node in chunking_subgraph_nodes:
            if node.op == "placeholder" and "tangent" in node.target:
                set_chunking_meta(node, scale_by=node)
                continue
        
            # weight of matmul is not chunked
            # assert all(get_chunking_meta(arg) is not None for arg in get_args_of_node_type(node)), f"{node.format_node()}"

            if node.op != "call_function":
                raise CantChunk("Chunker can only chunk call_function nodes")
            target = node.target
            if target not in propagate_rules:
                raise CantChunk(f"Missing propagation rule for target {target}: {node.format_node()}")

            if not propagate_rules[target](node):
                raise CantChunk(f"Propagate rule for {target} fail: {node.format_node()}")

# Begins propagation rules
@register_propagate_rule(aten.addmm.default)
def propagate_addmm(addmm_node):
    bias_node, input_node, weight_node = addmm_node.args

    # only input is chunked
    if get_chunking_meta(bias_node) is None and get_chunking_meta(weight_node) is None and get_chunking_meta(input_node) is not None:
        copy_chunking_meta(addmm_node, input_node)
        return True

    return False

@register_propagate_rule(aten.mm.default)
def propagate_mm(mm_node):
    lhs_node, rhs_node = mm_node.args[:2]
    lhs_meta = get_chunking_meta(lhs_node)
    rhs_meta = get_chunking_meta(rhs_node)

    # only lhs is chunked
    if lhs_meta is not None and rhs_meta is None:
        copy_chunking_meta(mm_node, lhs_meta)
        return True

    # both lhs and rhs are chunked at the reduction dimension
    if lhs_meta is not None and rhs_meta is not None and lhs_meta.chunk_dim == 1 and rhs_meta.chunk_dim == 0:
        # The output is not chunked, but need to be sum'ed up!
        scale_by = get_scale_by_from_metas(lhs_meta, rhs_meta)
        set_chunking_meta(mm_node, scale_by=scale_by, chunk_dim=None, need_sum=True)
        return True

    return False

@register_propagate_rule(aten.permute.default)
def propagate_permute(permute_node):
    input_node, order = permute_node.args[:2]
    input_meta = get_chunking_meta(input_node)
    if input_meta is None or input_meta.chunk_dim is None:
        return False

    orig_chunk_dim = input_meta.chunk_dim
    reverse_lookup = {v: k for k, v in enumerate(order)}
    new_chunk_dim = reverse_lookup[orig_chunk_dim]
    set_chunking_meta(permute_node, meta=input_meta, chunk_dim=new_chunk_dim)
    return True

@register_propagate_rule([
    aten.sub.Tensor,
    aten.add.Tensor,
])
def propagate_broadcastable(out_node):
    lhs_node, rhs_node = out_node.args[:2]

    lhs_meta = get_chunking_meta(lhs_node)

    lhs_ft = get_fake_tensor_from_node_arg(lhs_node)
    rhs_ft = get_fake_tensor_from_node_arg(rhs_node)
    if get_chunking_meta(rhs_node) is None and rhs_ft is not None and rhs_ft.ndim == 1 and lhs_ft is not None and lhs_ft.ndim > 1 and lhs_meta is not None and lhs_meta.chunk_dim is not None and lhs_meta.chunk_dim != lhs_ft.ndim - 1:
        copy_chunking_meta(out_node, lhs_node)
        return True

    return propagate_general_copy_from_input(out_node)

@register_propagate_rule([
    prims.convert_element_type.default,
    aten.exp.default,
    aten.log.default,
    aten.squeeze.dim,
    aten.gather.default,
    aten.neg.default,
    aten.scatter.value,
])
def propagate_general_copy_from_input(out_node, allow_non_chunked_scalar_input=False):
    """
    This rule assumes
    1. The node has at least one Node input
    2. Each Node input should have chunking meta
    3. Different nodes have the same chunking meta

    Then just copy the chunking meta to the output
    """
    node_args = get_args_of_node_type(out_node)

    if allow_non_chunked_scalar_input:
        new_args = []
        for arg in node_args:
            if compute_tensor_size(arg, count_bytes=False) == 1:
                if get_chunking_meta(arg) is not None:
                    return False
            else:
                new_args.append(arg)
        node_args = new_args
    if len(node_args) == 0:
        return False
    
    src_meta = get_chunking_meta(node_args[0])
    if src_meta is None:
        return False

    for other_node in node_args[1:]:
        other_meta = get_chunking_meta(other_node)
        if other_meta != src_meta:
            return False

    copy_chunking_meta(out_node, src_meta)
    return True

@register_propagate_rule([
    aten.where.self,
])
def propagate_where(where_node):
    # where_node can have a non-chunked scalar input
    if propagate_general_copy_from_input(where_node, True):
        return True

    # where(cond_node, true_node, false_node)
    # We can still propagate if
    # 1. cond_node is chunked
    # 2. true_node is not chunked but scaled
    # 3. false_node is not chunked and is always 0.
    #    It's important that the value is 0 since in that case
    #    scaling the output is always numerically safe.
    cond_node, true_node, false_node = where_node.args
    cond_meta = get_chunking_meta(cond_node)
    true_meta = get_chunking_meta(true_node)
    def can_chunk():
        # check false_node
        if get_chunking_meta(false_node) is not None:
            return False
        if false_node.target != aten.full.default:
            return False
        if false_node.meta["val"].numel() != 1:
            return False
        if false_node.args[1] != 0.0:
            return False

        # check true_node
        if true_meta.chunk_dim is not None or true_meta.need_sum:
            return False

        # check cond_meta
        if cond_meta.scale_by is not None or cond_meta.need_sum:
            return False
        
        return True

    if can_chunk():
        # the output is scaled if the input is scaled
        # the output is chunked if cond is chunked
        set_chunking_meta(where_node, meta=cond_meta, scale_by=true_meta.scale_by)
        return True
    return False

@register_propagate_rule(aten.div.Tensor)
def propagate_div(div_node):
    lhs_node, rhs_node = div_node.args[:2]
    lhs_meta = get_chunking_meta(lhs_node)
    rhs_meta = get_chunking_meta(rhs_node)

    if lhs_meta == rhs_meta:
        return propagate_general_copy_from_input(div_node)

    # Divide by a non-chunked scalar, just copy the metadata from
    # the numerator.
    if lhs_meta is not None and rhs_meta is None and rhs_node.meta["val"].numel() == 1:
        copy_chunking_meta(div_node, lhs_meta)
        return True

    return False


@register_propagate_rule([
    aten.sum.default,
])
def propagate_sum_to_scalar(sum_node):
    input_node = sum_node.args[0]
    input_meta = get_chunking_meta(input_node)
    assert input_meta

    # Input is not chunked
    if input_meta.chunk_dim is None:
        return False

    out_meta = ChunkingMeta(**dataclasses.asdict(input_meta))
    out_meta.need_sum = True
    out_meta.chunk_dim = None

    set_chunking_meta(sum_node, meta=out_meta)
    return True


@register_propagate_rule([
    aten.amax.default,
    aten.sum.dim_IntList,
])
def propagate_reduce_non_chunk_dim(reduce_node):
    """
    A reduction that reduces across non-chunked dimension.

    For sum, we also support reducing across the chunk dimension.
    """
    arg_node, reduce_dims = reduce_node.args[0: 2]
    arg_meta = get_chunking_meta(arg_node)
    
    if arg_meta is None:
        return False

    if arg_meta.chunk_dim not in reduce_dims:
        # Reduce across the none chunk dimension
        copy_chunking_meta(reduce_node, arg_node)
        return True

    if reduce_node.target == aten.sum.dim_IntList and list(reduce_dims) == [arg_meta.chunk_dim]:
        set_chunking_meta(reduce_node, scale_by=arg_meta.scale_by, chunk_dim=None, need_sum=True)
        return True
    
    return False

@register_propagate_rule([
    aten.full.default,
])
def propagate_full(full_node):
    if full_node.meta["val"].numel() != 1:
        set_chunking_meta(full_node, chunk_dim=0)
        return True
    return False

@register_propagate_rule([
    aten.expand.default,
])
def propagate_expand(expand_node):
    # expand from scalar
    if expand_node.meta["val"].numel() != 1:
        arg = expand_node.args[0]
        arg_meta = get_chunking_meta(arg)
        if arg_meta and arg.meta["val"].numel() == 1:
            set_chunking_meta(expand_node, chunk_dim=0, scale_by=arg_meta.scale_by)
            return True
    return False

def _propagate_mul(lhs_meta, rhs_meta):
    """
    Return out_meta if can chunk. Otherwise None.
    Having this API since this can be shared by aten.mul and aten.fma
    """
    if lhs_meta is None or rhs_meta is None:
        return None

    # Compare disregarding scale_by
    lhs_meta_copy = lhs_meta.copy()
    lhs_meta_copy.scale_by = None
    rhs_meta_copy = rhs_meta.copy()
    rhs_meta_copy.scale_by = None
    if lhs_meta_copy != rhs_meta_copy:
        return None

    # TODO: too restrictive. Loose the check if there is a use case
    # that both lhs and rhs are scaled.
    scale_by = get_scale_by_from_metas(lhs_meta, rhs_meta)

    out_meta = lhs_meta_copy.copy()
    out_meta.scale_by = scale_by
    return out_meta

@register_propagate_rule([
    aten.mul.Tensor,
])
def propagate_mul(mul_node):
    """
    We need special handling for `scale_by` for mul node.
    """
    lhs_node, rhs_node = mul_node.args[:2]
    lhs_meta = get_chunking_meta(lhs_node)
    rhs_meta = get_chunking_meta(rhs_node)
    out_meta = _propagate_mul(lhs_meta, rhs_meta)

    if out_meta is None:
        return False

    set_chunking_meta(mul_node, meta=out_meta)
    return True

@register_propagate_rule([
    prims.fma.default,
])
def propagate_fma(fma_node):
    lhs_node, rhs_node, addend_node = fma_node.args[:3]
    lhs_meta = get_chunking_meta(lhs_node)
    rhs_meta = get_chunking_meta(rhs_node)
    mul_meta = _propagate_mul(lhs_meta, rhs_meta)
    if mul_meta is None:
        return False

    addend_meta = get_chunking_meta(addend_node)
    if mul_meta == addend_meta:
        copy_chunking_meta(fma_node, mul_meta)
        return True
    return False

@register_propagate_rule([
    aten.view.default,
])
def propagate_view(view_node):
    """
    Only certain kinds of views can be chunked. E.g. if the view just removes some size==1 dimension.
    Or if the input node is not chunked.
    """

    arg = view_node.args[0]
    arg_meta = get_chunking_meta(arg)
    assert arg_meta is not None
    if arg_meta.chunk_dim is None:
        copy_chunking_meta(view_node, arg_meta)
        return True
    return False
