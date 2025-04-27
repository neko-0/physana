from typing import Union, Dict, List
import logging

from .container import BaseAnalysisContainer
from .region import Region

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class CutFlowNode(BaseAnalysisContainer):
    __slots__ = ("level", "selection", "total_nevents", "sumW", "sumW2")

    def __init__(self, name: str, selection: str):
        super().__init__(name)
        self.level: int = 0
        self.selection: str = selection
        self.total_nevents: int = 0
        self.sumW: float = 0
        self.sumW2: float = 0

    @property
    def effective_nevents(self) -> float:
        return self.sumW

    def connect(self, node: "CutFlowNode") -> None:
        if not isinstance(node, CutFlowNode):
            raise TypeError("node must be an instance of CutFlowNode")
        self._data[node._name] = node
        node.level += 1 + self.level
        node.parent = self

    def clear(self) -> None:
        self.total_nevents = 0
        self._sumW = 0.0
        self._sumW2 = 0.0
        for node in self:  # same as self._data.values()
            node.clear()

    def chain_selection(self) -> str:
        cuts = [self.selection]
        parent = self.parent
        while parent:
            cuts.append(parent.selection)
            parent = parent.parent
        return " && ".join(cuts)


def create_cutflow_chain(cut_dict: Dict[str, str]) -> Dict[str, CutFlowNode]:
    nodes = {name: CutFlowNode(name, cut) for name, cut in cut_dict.items()}
    for name, node in nodes.items():
        # Create a new top-level chain for each node
        remaining_nodes = [
            CutFlowNode(n._name, n.selection)
            for n in nodes.values()
            if n._name != node._name
        ]

        # Recursively create the chain
        def create_chain(
            current_node: CutFlowNode,
            node_list: List[CutFlowNode],
            prefix: str = "None",
        ) -> None:
            logger.debug(f"{prefix} -> {[node._name for node in node_list]}")
            for other_node in node_list:
                current_node.connect(other_node)
                logger.debug(f"connected {other_node._name}")
            for other_node in node_list:
                next_node_list = [
                    CutFlowNode(x._name, x.selection)
                    for x in node_list
                    if x._name != other_node._name
                ]
                if next_node_list:
                    create_chain(
                        other_node,
                        next_node_list,
                        prefix=f"{prefix} -> {other_node._name}",
                    )

        logger.debug(
            f"starting {name}, connecting {[node._name for node in remaining_nodes]}"
        )
        create_chain(node, remaining_nodes, prefix=name)

    return nodes


class RegionCutFlow(Region):
    __slots__ = ("selection_dict", "cutflow")

    def __init__(
        self,
        name: str,
        weights: Union[str, List[str]],
        selection_dict: Dict[str, str],
    ):
        super().__init__(name, weights, "&&".join(selection_dict.values()))
        self.selection_dict: Dict[str, str] = selection_dict
        self.cutflow: Dict[str, CutFlowNode] = create_cutflow_chain(selection_dict)

    def append(self, *args, **kwargs) -> None:
        pass

    def clear_content(self) -> None:
        pass
