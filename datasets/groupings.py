"""Contains groupings for experiments"""

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(init=False)
class Grouping(object):
    """Instantiates one grouping"""

    def __init__(self, task_label: str, subgroup_attribute: List[str]):
        self.task_labels = [task_label]
        self.subgroup_attributes = {task_label: subgroup_attribute}


@dataclass(init=False)
class MTLGrouping(object):
    """Instantiates multiple groupings"""

    def __init__(self, *groupings: Iterable[Grouping]):
        task_labels = []
        subgroup_attributes = {}
        for grouping in groupings:
            assert isinstance(grouping, Grouping)
            task_labels.extend(grouping.task_labels)
            subgroup_attributes.update(grouping.subgroup_attributes)
        self.task_labels = task_labels
        self.subgroup_attributes = subgroup_attributes


single_task_groupings = {
    # DEFAULT (yes, this is also grouping #7)
    0: Grouping(task_label="Blond_Hair", subgroup_attribute=["Male"]),

    # Grouping 1: Disjoint spurious correlates
    1: Grouping(task_label="Attractive", subgroup_attribute=["Eyeglasses"]),
    2: Grouping(task_label="Smiling", subgroup_attribute=["High_Cheekbones"]),
    3: Grouping(task_label="Young", subgroup_attribute=["Attractive"]),
    # 2: Grouping(task_label="Smiling", subgroup_attribute=["High_Cheekbones"]),

    # Grouping 2: Non-disjoint spurious correlates
    4: Grouping(task_label="Oval_Face", subgroup_attribute=["Rosy_Cheeks"]),
    5: Grouping(task_label="Attractive", subgroup_attribute=["Bald"]),
    6: Grouping(task_label="Young", subgroup_attribute=["Gray_Hair"]),
    7: Grouping(task_label="Blond_Hair", subgroup_attribute=["Male"]),
    8: Grouping(task_label="Pointy_Nose", subgroup_attribute=["Male"]),

    # Grouping 3: Similary bad ERM models
    9: Grouping(task_label="Pointy_Nose", subgroup_attribute=["Rosy_Cheeks"]),
    10: Grouping(task_label="Attractive", subgroup_attribute=["Heavy_Makeup"]),
}


def get_grouping(indices):
    return MTLGrouping(*[single_task_groupings[index] for index in indices])
