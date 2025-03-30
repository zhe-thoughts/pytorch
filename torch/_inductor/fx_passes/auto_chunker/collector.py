"""
Collect the source node to chunk and all the nodes that should be put in
the chunking subgraph.
"""

import torch
from torch.utils._pytree import tree_flatten
from typing import Sequence, Set
from torch.fx import Graph, Node
from torch._inductor import config
import itertools
from .utils import get_args_of_node_type, use_tangent, get_fake_tensor_from_node_arg, compute_tensor_size
from .core import CantChunk, find_amplifier_node

aten = torch.ops.aten
prims = torch.ops.prims


def print_non_selected_nodes(graph: Graph, filter_set: Set):
    print("Remaining nodes:")
    for idx, node in enumerate(graph.nodes):
        if node not in filter_set:
            fake_tensor = get_fake_tensor_from_node_arg(node)
            shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
            print(f"  {idx:3}: {shape} {node.format_node()}")

def get_tangent_nodes(graph: Graph):
    tangents = []
    for node in graph.nodes:
        if node.op == "placeholder" and "tangent" in node.name:
            tangents.append(node)
    return tangents


def get_tangent_node(graph: Graph) -> Node:
    """
    Return the single tangent node. Raise CantChunk if the graph has
    more than one tangents.
    """
    tangents = get_tangent_nodes(graph)
    if len(tangents) != 1:
        raise CantChunk("Can chunk only if there is a single tangent")
    return tangents[0]


# Right now only allow matmul/addmm as the source nodes for chunking.
# May extend it to more ops.
# The index points to the argument of the node that should be chunked.
eligible_source_node_op_to_idx = {
    aten.mm.default: 0,
    # arg[0] is the bias. We should chunk arg[1] which is the input
    # matrix for the matmul.
    aten.addmm.default: 1,
}

def maybe_permuted(downstream_node, upstream_node):
    """
    Return true if downstream_node is upstream_node or a permutation of it
    """

    if downstream_node is upstream_node:
        return True

    if downstream_node.target == aten.permute.default and downstream_node.args[0] is upstream_node:
        return True
    return False


def node_is_returned(node):
    return any(n.op == "output" for n in node.users)


class Collector:
    @classmethod
    def collect_source_user(cls, graph: Graph):
        """ 
        Pick the one with the largest amplification factor for now.
        But an alternative reasonable strategy is to pick the one
        closest to the bwd part
        """
        amplifier_node = find_amplifier_node(graph)
        if amplifier_node is None:
            raise CantChunk("No source nodes found.")

        return amplifier_node

    @classmethod
    def collect_source_node(cls, graph: Graph):
        source_user = cls.collect_source_user(graph)
        argidx = eligible_source_node_op_to_idx[source_user.target]
        return source_user.args[argidx]

    @classmethod
    def _find_reachable_nodes(cls, start_node, can_expand = lambda _: True):
        """
        Collect the nodes reachable from the `start_node` via the use chain.
        The output is topologically sorted.
        """
        reachable_nodes = []
        seen = set()
    
        def dfs(node):
            if node in seen:
                return
    
            seen.add(node)
    
            # no need to add the output node since the output node should not
            # see the effect of chunking.
            if node.op == "output":
                return
    
            if can_expand(node):
                for node2 in node.users:
                    dfs(node2)
    
                reachable_nodes.append(node)
            else:
                # If we are at leaves of expansion, only add the node if
                # it's not a view op
                if node.target != aten.view.default:
                    reachable_nodes.append(node)
    
        dfs(start_node)
    
        return list(reversed(reachable_nodes))

    @classmethod
    def find_reachable_nodes_backward(cls, chunking_subgraph_nodes, candidate_list, source_node, source_user):
        """
        Discover node as large as `source_user` from the `candidate_list`.
        """
        source_user_size = compute_tensor_size(source_user, count_bytes=False)
        while candidate_list:
            curr_node = candidate_list.pop()
            curr_node_size = compute_tensor_size(curr_node, count_bytes=False)

            if curr_node_size == source_user_size:
                chunking_subgraph_nodes.add(curr_node)
                candidate_list += get_args_of_node_type(curr_node)
            else:
                for arg in get_args_of_node_type(curr_node):
                    # If an op uses `source_node`, we add it
                    if arg is source_node:
                        chunking_subgraph_nodes.add(curr_node)
                        break

    @classmethod
    def collect_chunking_subgraph_nodes(cls, graph: torch.fx.Graph) -> Set[Node]:
        source_user = cls.collect_source_user(graph)
        argidx = eligible_source_node_op_to_idx[source_user.target]
        source_node = source_user.args[argidx]

        # TODO This check is specific to matmul/addmm. Need generalize
        # to support other kind of `source_node`
        def _can_expand(node):
            if node_is_returned(node):
                return False

            if node.target != aten.mm.default:
                return True
            
            if node == source_user:
                return True
           
            argidx = eligible_source_node_op_to_idx[source_user.target]
            source_matmul_inputs = source_user.args[argidx : argidx + 2]
            assert len(node.args) == 2
            cur_node_inputs = node.args

            for downstream_node, upstream_node in itertools.product(cur_node_inputs, source_matmul_inputs):
                if maybe_permuted(downstream_node, upstream_node):
                    # compute gradient of input/weight. Stop expanding
                    return False
            return True

        # The chunking subgraph should contains nodes reachable from
        # the source_ndoe
        chunking_subgraph_nodes = set(cls._find_reachable_nodes(source_user, _can_expand))

        # The chunking subgraph should also contains nodes reachable from
        # tangents.
        # TODO: this is specific for training now. Need revise if we have
        # inference usecase.
        tangent = get_tangent_node(graph)
        if tangent.meta["val"].numel() != 1:
            raise CantChunk("The graph does not have a single scalar gradient.")

        last_fwd_node = next(iter(tangent.users))._prev
        if last_fwd_node not in chunking_subgraph_nodes:
            raise CantChunk("Can only chunk all the way to the end of the graph")
        chunking_subgraph_nodes |= set(cls._find_reachable_nodes(tangent,
            _can_expand))

        # The last part of nodes for the chunking subgraph are a bit hard
        # to find.
        #
        # The gather op for CrossEntropyLoss in the forward will cause an
        # scatter op in the backward. The scatter op runs on the output of
        # aten.full. This `aten.full` is also the tensor we need to chunk. Otherwise,
        # this tensor need to be passed into the chunking subgraph as inputs
        # which costs memory.
        #
        # We discover such nodes by check all nodes used as args in
        # `chunking_subgraph_nodes` but itself is not in `chunking_subgraph_nodes`.
        # We check if such node can go backward in the graph and reach a node which is as large as the
        # `chunk_user` node. Add those nodes to `chunking_subgraph_nodes`
        external_args = set(itertools.chain.from_iterable(get_args_of_node_type(node) for node in chunking_subgraph_nodes)) - chunking_subgraph_nodes
        cls.find_reachable_nodes_backward(chunking_subgraph_nodes, list(external_args), source_node, source_user)

        # print_non_selected_nodes(graph, chunking_subgraph_nodes)
        return chunking_subgraph_nodes
