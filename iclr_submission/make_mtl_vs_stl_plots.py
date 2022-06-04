import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


mtl_results = {}
stl_results = {}

stl_results["ERM"] = {"Big_Lips:Chubby": [(69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.6, (92.39,92.81)), (13.64, (5.36,21.92)), (90.8, (90.57,91.03)), (27.04, (16.94,37.15))]
        "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
        "Gray_Hair:Young": [(98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))],
        "High_Cheekbones:Smiling":
        [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))]}

stl_results["RWY"] = {"Big_Lips:Chubby": [(69.58, (69.22,69.95)), (49.31, (48.59,50.03)), (67.44, (67.07,67.82)), (57.85, (57.14,58.57))],
    "Bushy_Eyebrows:Blond_Hair": [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))],
    "Wearing_Lipstick:Male":
        [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))],
        "Gray_Hair:Young": [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))],
        "High_Cheekbones:Smiling":
        [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))]}

stl_results["SUBY"] = {"Big_Lips:Chubby": [(69.46, (69.09,69.83)), (28.6, (27.95,29.25)), (68.69, (68.32,69.06)), (60.39, (59.69,61.1))],
     "Bushy_Eyebrows:Blond_Hair": [(91.31, (91.08,91.54)), (16.17, (7.54,24.8)), (77.0, (76.66,77.34)), (71.64, (71.22,72.06))],
     "Wearing_Lipstick:Male":
        [(93.61, (93.42,93.81)), (35.23, (27.37,43.08)), (93.15, (92.95,93.36)), (38.88, (30.86,46.9))],
        "Gray_Hair:Young": [(95.93, (95.78,96.09)), (41.15, (30.53,51.76)), (93.0, (92.79,93.2)), (74.1, (73.17,75.03))],
        "High_Cheekbones:Smiling":
        [(87.67, (87.4,87.93)), (38.49, (37.04,39.95)), (85.13, (84.84,85.41)), (51.95, (50.57,53.33))]}

stl_results["JTT"] = {"Big_Lips:Chubby": [(32.7, (32.33,33.08)), (0.0, (0.0,0.0)), (32.7, (32.33,33.08)), (0.0, (0.0,0.0))],
    "Bushy_Eyebrows:Blond_Hair": [(91.94, (91.72,92.16)), (21.04, (11.22,30.86)), (71.45, (71.09,71.81)), (64.15, (63.71,64.6))],
    "Wearing_Lipstick:Male":
        [(91.27, (91.05,91.5)), (42.45, (34.32,50.57)), (72.96, (72.6,73.32)), (51.52, (49.32,53.73))],
        "Gray_Hair:Young": [(97.81, (97.69,97.93)), (22.31, (12.82,31.8)), (90.83, (90.6,91.06)), (68.34, (67.53,69.14))],
        "High_Cheekbones:Smiling":
        [(TODO, (TODO)), (TODO, (TODO)), (TODO, (TODO), (TODO, (TODO))]}


mtl_results["ERM"] = {
    "Big_Lips:Chubby": [(70.6, (70.24, 70.97)), (15.49, (14.97, 16.02)), (66.9, (66.53, 67.28)), (58.84, (58.23, 59.45))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.76, (92.55, 92.96)), (14.2, (6.02, 22.39)), (88.52, (88.27, 88.78)), (41.41, (29.86, 52.96))]
}

mtl_results["RWY"] = {
    "Big_Lips:Chubby": [(69.63, (69.26,70.0)), (29.8, (29.14,30.45)), (66.05, (65.68,66.43)), (58.84, (58.13,59.55))],
    "Bushy_Eyebrows:Blond_Hair":
        [(88.31, (88.05,88.56)), (44.27, (32.62,55.92)), (80.63, (80.31,80.95)), (73.52, (72.8,74.23))]
}

mtl_results["SUBY"] = {
    "Big_Lips:Chubby": [(67.32, (66.95,67.7)), (47.35, (46.63,48.07)), (64.06, (63.68,64.45)), (61.44, (60.84,62.04))],
    "Bushy_Eyebrows:Blond_Hair":
        [(84.35, (84.06,84.65)), (45.7, (34.02,57.39)), (79.48,(79.15,79.8)), (72.22, (71.5,72.95))]
}

mtl_results["JTT"] = {
    "Big_Lips:Chubby": [(61.1, (60.71,61.49)), (53.97, (53.42,54.52)), (59.93, (59.54,60.33)), (50.46, (49.9,51.01))],
    "Bushy_Eyebrows:Blond_Hair":
        [(72.9, (72.55,73.26)), (67.22, (66.79,67.66)), (77.24, (76.91,77.58)), (68.85, (68.1,69.6))]
}


def make_plot(wg_acc = True):

    plt.figure(figsize=(12, 7))
    avg_opt_data = []

    y_label = "Mean Worst Group Accuracy Across Tasks" if wg_acc else "Mean Average Accuracy Across Tasks"

    for opt_type in mtl_results.keys():
        
        avg_mtl = 0
        avg_stl = 0

        for task_name in mtl_results[opt_type]:

            mtl_res = mtl_results[opt_type][task_name]
            stl_res = stl_results[opt_type][task_name]

            if wg_acc:
                avg_mtl += mtl_res[3][0]
                avg_stl += stl_res[3][0]
            else:
                avg_mtl += mtl_res[2][0]
                avg_stl += stl_res[2][0]
                

        avg_mtl /= len(mtl_results[opt_type])
        avg_stl /= len(mtl_results[opt_type])

        avg_opt_data.append([avg_stl, "STL", opt_type])
        avg_opt_data.append([avg_mtl, "MTL", opt_type])


    avg_opt_df = pd.DataFrame(avg_opt_data, columns=[y_label, 'Learning Type', 'Optimization Procedure'])


    sns.barplot(x = 'Optimization Procedure',
            y = y_label,
            hue = 'Learning Type',
            data = avg_opt_df)

            
    plt.title(f"STL vs MTL Comparison")
    plt.savefig(f"./plots/stl_mtl_comparison_wg_{wg_acc}.png")
    plt.close()

make_plot(wg_acc = True)
make_plot(wg_acc = False)