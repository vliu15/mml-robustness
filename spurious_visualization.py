import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

## for all files in the reultls of a given log dir, load them into dicts, put into pandas and then display it

task_label = "Smiling"
attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]
attributes.remove(task_label)
log_dir = './logs/erm_smiling'

results_dict = {"Group 0 Acc": [], "Group 1 Acc": [], "Group 2 Acc": [], "Group 3 Acc": []}

for attr in attributes:

    results_dir = os.path.join(log_dir, "results_best_val")
    file = os.path.join(results_dir, f"{task_label}_{attr}_test_results.json")

    attr_dict = json.load(open(file))

    for i in range(4):
        results_dict[f'Group {i} Acc'].append(attr_dict[f'{task_label}_g{i}'])

results_df = pd.DataFrame.from_dict(results_dict)
results_df = results_df.rename(index={ind: v for ind, v in enumerate(attributes)})

fig, ax = plt.subplots(figsize=(16, 10))
heatmap = sns.heatmap(results_df, annot=True, linewidths=.5, ax=ax)
ax.set_title(f'Task Label: {task_label}')
plt.xlabel("Subgroups")
plt.ylabel("Potential Spurrious Correlates")
heatmap.figure.savefig(os.path.join(results_dir, "heatmap.png"))
## put into sns

## svae sns image
