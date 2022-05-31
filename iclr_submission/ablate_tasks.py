import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

stl_erm_disjoint = {
    "Big_Lips:Chubby": [(69.93, (69.57, 70.30)), (11.27, (10.81, 11.72)), (68.32, (67.94, 68.69)), (41.26, (40.55, 41.97))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.60, (92.39, 92.81)), (15.64, (7.12, 24.15)), (90.62, (90.39, 90.85)), (32.82, (21.81, 43.83))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.29, 93.69)), (37.23, (29.35, 45.10)), (92.67, (92.46, 92.88)), (49.65, (41.51, 57.80))],
    "Gray_Hair:Young": [(98.15, (98.05, 98.26)), (26.27, (16.36, 36.17)), (97.94, (97.83, 98.06)), (18.36, (9.64, 27.07))],
    "High_Cheekbones:Smiling":
        [(87.39, (87.12, 87.65)), (38.26, (36.81, 39.72)), (84.29, (84.00, 84.58)), (50.03, (48.6, 51.47))],
    "Brown_Hair:Wearing_Hat":
        [(88.67, (88.42, 88.93)), (24.34, (15.04, 33.64)), (86.48, (86.21, 86.75)), (39.0, (28.44, 49.57))],
    "No_Beard:Wearing_Lipstick":
        [(96.06, (95.91, 96.22)), (26.12, (6.28, 45.95)), (94.68, (94.5, 94.86)), (10.19, (-3.47, 23.86))],
    "Young:Chubby": [(88.01, (87.75, 88.27)), (63.12, (62.26, 63.97)), (85.94, (85.66, 86.21)), (66.56, (65.55, 67.57))],
    "Bangs:Wearing_Hat": [(95.89, (95.73, 96.05)), (59.04, (46.68, 71.4)), (95.36, (95.19, 95.53)), (68.9, (57.27, 80.53))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.42, (77.08, 77.75)), (20.47, (19.5, 21.44)), (71.73, (71.37, 72.09)), (39.76, (38.58, 40.93))]
}

mtl_erm_ablate = {
    2:
        {
            "Big_Lips:Chubby":
                [(70.6, (70.24, 70.97)), (15.49, (14.97, 16.02)), (66.9, (66.53, 67.28)), (58.84, (58.23, 59.45))],
            "Bushy_Eyebrows:Blond_Hair":
                [(92.76, (92.55, 92.96)), (14.2, (6.02, 22.39)), (88.52, (88.27, 88.78)), (41.41, (29.86, 52.96))],
        },
    3:
        {
            "Big_Lips:Chubby":
                [(70.57, (70.2, 70.93)), (13.76, (13.26, 14.26)), (68.29, (67.92, 68.67)), (50.24, (49.52, 50.96))],
            "Bushy_Eyebrows:Blond_Hair":
                [(92.46, (92.24, 92.67)), (14.2, (6.02, 22.39)), (89.48, (89.24, 89.73)), (42.84, (31.24, 54.45))],
            "Wearing_Lipstick:Male":
                [(93.63, (93.43, 93.82)), (35.16, (27.38, 42.93)), (93.39, (93.19, 93.59)), (41.37, (33.35, 49.39))],
        },
    4:
        {
            "Big_Lips:Chubby":
                [(71.21, (70.85, 71.58)), (17.61, (17.06, 18.16)), (66.92, (66.54, 67.29)), (53.63, (52.91, 54.35))],
            "Bushy_Eyebrows:Blond_Hair":
                [(92.7, (92.49, 92.91)), (15.64, (7.12, 24.15)), (89.45, (89.2, 89.69)), (39.98, (28.49, 51.47))],
            "Wearing_Lipstick:Male":
                [(93.93, (93.74, 94.13)), (31.7, (24.13, 39.28)), (93.69, (93.5, 93.89)), (39.99, (32.01, 47.97))],
            "Gray_Hair:Young": [
                (98.21, (98.1, 98.31)), (28.9, (18.7, 39.11)), (97.55, (97.43, 97.67)), (42.09, (30.98, 53.2))
            ],
        },
    5:
        {
            "Big_Lips:Chubby":
                [(71.8, (71.43, 72.16)), (20.88, (20.29, 21.47)), (67.15, (66.77, 67.52)), (53.51, (52.79, 54.23))],
            "Bushy_Eyebrows:Blond_Hair":
                [(92.75, (92.54, 92.95)), (12.77, (4.94, 20.6)), (89.57, (89.33, 89.82)), (44.27, (32.62, 55.92))],
            "Wearing_Lipstick:Male":
                [(93.94, (93.75, 94.13)), (31.7, (24.13, 39.28)), (93.32, (93.12, 93.52)), (35.16, (27.38, 42.93))],
            "Gray_Hair:Young":
                [(98.14, (98.03, 98.25)), (26.27, (16.36, 36.17)), (97.68, (97.56, 97.8)), (30.22, (19.89, 40.56))],
            "High_Cheekbones:Smiling":
                [(87.83, (87.57, 88.1)), (37.7, (36.25, 39.15)), (84.4, (84.11, 84.69)), (51.21, (49.88, 52.55))],
        },
    6:
        {
            "Big_Lips:Chubby":
                [(70.73, (70.37, 71.1)), (14.14, (13.64, 14.64)), (67.43, (67.06, 67.81)), (53.91, (53.19, 54.63))],
            "Bushy_Eyebrows:Blond_Hair":
                [(92.63, (92.43, 92.84)), (14.2, (6.02, 22.39)), (88.55, (88.29, 88.8)), (41.41, (29.86, 52.96))],
            "Wearing_Lipstick:Male":
                [(93.6, (93.4, 93.79)), (34.47, (26.73, 42.21)), (93.44, (93.24, 93.64)), (43.44, (35.37, 51.51))],
            "Gray_Hair:Young":
                [(98.17, (98.06, 98.28)), (27.58, (17.53, 37.64)), (97.16, (97.03, 97.29)), (44.73, (33.54, 55.92))],
            "High_Cheekbones:Smiling":
                [(87.69, (87.42, 87.95)), (39.03, (37.57, 40.49)), (85.57, (85.29, 85.85)), (52.27, (50.94, 53.6))],
            "Brown_Hair:Wearing_Hat":
                [(88.57, (88.31, 88.82)), (17.01, (8.87, 25.15)), (85.62, (85.34, 85.9)), (47.56, (36.74, 58.38))],
        },
}


def task_ablation_average_performances(tasks_to_compare=set(["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"])):
    """Computes the individual average task performances from MTL and STL"""
    num_tasks_to_avg_acc = {}
    num_tasks_to_wg_acc = {}

    # Compute average of STL ERM
    avg_acc = []
    wg_acc = []
    for task, metrics in stl_erm_disjoint.items():
        if task in tasks_to_compare:
            _, _, avg_from_wg, wg_from_wg = metrics
            # avg_from_wg, wg_from_wg, _, _ = metrics
            avg_acc += [avg_from_wg[0]]
            wg_acc += [wg_from_wg[0]]
    avg_acc = sum(avg_acc) / len(avg_acc)
    wg_acc = sum(wg_acc) / len(wg_acc)

    num_tasks_to_avg_acc[1] = avg_acc
    num_tasks_to_wg_acc[1] = wg_acc

    # Compute average avg/wg accuracies across the tasks in each task ablation
    for num_tasks, tasks in mtl_erm_ablate.items():
        if tasks_to_compare.issubset(set(tasks.keys())):
            avg_acc = []
            wg_acc = []
            for task, metrics in tasks.items():
                if task in tasks_to_compare:
                    _, _, avg_from_wg, wg_from_wg = metrics
                    # avg_from_wg, wg_from_wg, _, _ = metrics
                    avg_acc += [avg_from_wg[0]]
                    wg_acc += [wg_from_wg[0]]
            num_tasks_to_avg_acc[num_tasks] = sum(avg_acc) / len(avg_acc)
            num_tasks_to_wg_acc[num_tasks] = sum(wg_acc) / len(wg_acc)

    x = list(sorted(num_tasks_to_avg_acc.keys()))
    data = pd.DataFrame(
        {
            "# Tasks": x,
            "Average of avg acc": [num_tasks_to_avg_acc[k] for k in x],
            "Average of worst-group acc": [num_tasks_to_wg_acc[k] for k in x],
        }
    )

    ax = sns.lineplot(
        x="# Tasks",
        y="value",
        hue="variable",
        data=pd.melt(data, id_vars=["# Tasks"], value_vars=["Average of avg acc", "Average of worst-group acc"])
    )
    ax.set_title(f"Tasks: {list(tasks_to_compare)}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("./ablate_tasks.png")


if __name__ == "__main__":
    # task_ablation_average_performances(tasks_to_compare=set(["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]))
    # task_ablation_average_performances(tasks_to_compare=set(["Big_Lips:Chubby"]))
    # task_ablation_average_performances(tasks_to_compare=set(["Bushy_Eyebrows:Blond_Hair"]))
    # task_ablation_average_performances(tasks_to_compare=set(["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male"]))
    task_ablation_average_performances(tasks_to_compare=set(["Wearing_Lipstick:Male"]))
