"""Contains groupings for experiments"""

from dataclasses import dataclass
from typing import Iterable, List

ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


@dataclass(init=False)
class Grouping(object):
    """Instantiates one grouping"""

    def __init__(self, task_label: str, subgroup_attributes: List[str]):
        self.task_labels = [task_label]
        self.subgroup_attributes = {task_label: subgroup_attributes}


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


# DEPRECATED
# default_groupings = {
#     # DEFAULT (yes, this is also grouping #7)
#     0: Grouping(task_label="Blond_Hair", subgroup_attributes=["Male"]),

#     # Grouping 1: Disjoint spurious correlates
#     1: Grouping(task_label="Attractive", subgroup_attributes=["Eyeglasses"]),
#     2: Grouping(task_label="Smiling", subgroup_attributes=["High_Cheekbones"]),
#     3: Grouping(task_label="Young", subgroup_attributes=["Attractive"]),
#     # 2: Grouping(task_label="Smiling", subgroup_attributes=["High_Cheekbones"]),

#     # Grouping 2: Non-disjoint spurious correlates
#     4: Grouping(task_label="Oval_Face", subgroup_attributes=["Rosy_Cheeks"]),
#     5: Grouping(task_label="Attractive", subgroup_attributes=["Bald"]),
#     6: Grouping(task_label="Young", subgroup_attributes=["Gray_Hair"]),
#     7: Grouping(task_label="Blond_Hair", subgroup_attributes=["Male"]),
#     8: Grouping(task_label="Pointy_Nose", subgroup_attributes=["Male"]),

#     # Grouping 3: Similary bad ERM models
#     9: Grouping(task_label="Pointy_Nose", subgroup_attributes=["Rosy_Cheeks"]),
#     10: Grouping(task_label="Attractive", subgroup_attributes=["Heavy_Makeup"]),
# }


def get_grouping_object(tasks):
    """Takes in grouping string and returns a Python dataclass"""
    groupings = []
    for task in tasks:
        task_label, subgroup_attributes = task.split(":")
        subgroup_attributes = subgroup_attributes.split(",")
        subgroup_attributes = list(set(subgroup_attributes))
        groupings += [Grouping(task_label=task_label, subgroup_attributes=subgroup_attributes)]
    return MTLGrouping(*groupings)
