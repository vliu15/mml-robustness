"""Generates all MTL task ablation plots."""

import argparse
import logging
import logging.config
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


stl_erm_disjoint = {
    "Big_Lips:Chubby": [(69.94, (69.57,70.3)), (11.01, (10.56,11.46)), (68.32, (67.94,68.69)), (41.24, (40.53,41.95))],
    "Bushy_Eyebrows:Blond_Hair":
        [(92.6, (92.39,92.81)), (13.64, (5.36,21.92)), (90.8, (90.57,91.03)), (27.04, (16.94,37.15))],
    "Wearing_Lipstick:Male":
        [(93.49, (93.3,93.69)), (36.72, (28.78,44.65)), (92.68, (92.47,92.89)), (49.66, (41.63,57.7))],
    "Gray_Hair:Young": [(98.16, (98.05,98.26)), (25.0, (15.0,35.0)), (97.95, (97.84,98.06)), (15.32, (7.09,23.55))],
    "High_Cheekbones:Smiling":
        [(87.39, (87.12,87.65)), (38.24, (36.79,39.69)), (84.3, (84.01,84.6)), (50.03, (48.6,51.47))],
    "Brown_Hair:Wearing_Hat":
        [(88.68, (88.42,88.93)), (22.77, (13.49,32.05)), (86.49, (86.21,86.76)), (38.46, (27.66,49.26))],
    "No_Beard:Wearing_Lipstick":
        [(96.05, (95.9,96.21), (20.0, (0.0,40.24)), (94.65, (94.47,94.83)), (0.0, (0.0,0.0))],
    "Young:Chubby": [(88.01, (87.75,88.27)), (63.12, (62.26,63.98)), (85.97, (85.69,86.25)), (66.57, (65.56,67.58))],
    "Bangs:Wearing_Hat": [(95.93, (95.77,96.08)), (63.16, (50.64,75.68)), (95.55, (95.38,95.71)), (77.36, (66.71,88.01))],
    "Pointy_Nose:Heavy_Makeup":
        [(77.34, (77.01,77.68)), (20.21, (19.25,21.18)), (72.46, (72.1,72.81)), (35.24, (34.09,36.39))]
}

stl_erm_full_task_ablation = {
    "Arched_Eyebrows:Male":
        [(83.79, (83.5,84.09)), (35.0, (32.41,37.58)), (80.62, (80.3,80.93)), (50.47, (47.77,53.18))],
    "Big_Nose:Male": [(83.98, (83.69,84.28)), (35.18, (33.68,36.68)), (79.52, (79.2,79.85)), (46.68, (45.12,48.24))],
    "Blond_Hair:Male": [(95.8, (95.64,95.97)), (42.58, (38.41,46.75)), (95.18, (95.01,95.35)), (43.33, (39.15,47.5))],
    "Wearing_Earrings:Male":
        [(90.38, (90.14,90.62)), (32.67, (28.74,36.6)), (88.66, (88.41,88.92)), (42.28, (38.14,46.42))] 
}

stl_erm_nondisjoint = {
    "Wearing_Earrings:Male": [
        (90.38, (90.14,90.62)), (32.67, (28.74,36.6)), (88.66, (88.41,88.92)), (42.28, (38.14,46.42)) 
    ],
    "Attractive:Male": [
        (82.18, (81.87,82.48)),
        (66.11, (65.21,67.02)),
        (79.76, (79.44,80.08)),
        (65.62, (64.39,66.85)),
    ],
    "No_Beard:Heavy_Makeup": [
        (96.05, (95.9,96.21)),
        (28.99, (12.0,45.98)),
        (89.7, (89.45,89.94)),
        (0.0, (0.0,0.0)),
    ],
    "Pointy_Nose:Heavy_Makeup":
        [
            (77.34, (77.01,77.68)),
            (20.21, (19.25,21.18)),
            (72.46, (72.1,72.81)),
            (35.24, (34.09,36.39)),
        ],
    "Attractive:Gray_Hair": [
        (82.2, (81.9,82.51)),
        (38.69, (25.76,51.63)),
        (80.67, (80.35,80.98)),
        (50.0, (36.72,63.28)),
    ],
    "Big_Nose:Gray_Hair": [
        (84.0, (83.71,84.29)),
        (53.86, (52.77,54.95)),
        (80.52, (80.21,80.84)),
        (56.93, (55.48,58.39)),
    ],
    "Heavy_Makeup:Wearing_Lipstick":
        [(91.34, (91.11,91.56)), (31.19, (26.76,35.62)), (90.58, (90.34,90.81)), (33.26, (28.76,37.77))],
    "No_Beard:Wearing_Lipstick":
        [
            (96.05, (95.9,96.21)),
            (20.0, (0,40.24)),
            (94.65, (94.47,94.83)),
            (0.0, (0.0,0.0)),
        ],
    "Bangs:Wearing_Hat": [
        (95.93, (95.77,96.08)),
        (63.16, (50.64,75.68)),
        (95.55, (95.38,95.71)),
        (77.36, (66.71,88.01)),
    ],
    "Blond_Hair:Wearing_Hat": [
        (95.86, (95.7,96.02)), (35.71, (20.72,50.71)), (94.93, (94.76,95.11)), (48.72, (33.07,64.36))
    ]
}

mtl_erm_disjoint_ablate = {
    2:
        [
            {
                "Big_Lips:Chubby":
                    [(70.6, (70.24, 70.97)), (15.49, (14.97, 16.02)), (66.9, (66.53, 67.28)), (58.84, (58.23, 59.45))],
                "Bushy_Eyebrows:Blond_Hair":
                    [(92.76, (92.55, 92.96)), (14.2, (6.02, 22.39)), (88.52, (88.27, 88.78)), (41.41, (29.86, 52.96))],
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (12.77, (4.94, 20.6)), (0., (0., 0.)), (44.27, (32.62, 55.92))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (38.01, (36.55, 39.46)), (0., (0., 0.)), (51.9, (50.46, 53.33))],
            }
        ],
    3:
        [
            {
                "Big_Lips:Chubby":
                    [(70.57, (70.2, 70.93)), (13.76, (13.26, 14.26)), (68.29, (67.92, 68.67)), (50.24, (49.52, 50.96))],
                "Bushy_Eyebrows:Blond_Hair":
                    [(92.46, (92.24, 92.67)), (14.2, (6.02, 22.39)), (89.48, (89.24, 89.73)), (42.84, (31.24, 54.45))],
                "Wearing_Lipstick:Male":
                    [(93.63, (93.43, 93.82)), (35.16, (27.38, 42.93)), (93.39, (93.19, 93.59)), (41.37, (33.35, 49.39))],
            }, {
                "Big_Lips:Chubby": [(0., (0., 0.)), (15.56, (15.04, 16.08)), (0., (0., 0.)), (57.57, (56.96, 58.18))],
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (12.77, (4.94, 20.6)), (0., (0., 0.)), (27.09, (16.67, 37.51))],
                "Gray_Hair:Young": [(0., (0., 0.)), (28.9, (18.7, 39.11)), (0., (0., 0.)), (39.45, (28.45, 50.45))]
            }, {
                "Big_Lips:Chubby": [(0., (0., 0.)), (17.47, (16.92, 18.02)), (0., (0., 0.)), (53.64, (52.92, 54.36))],
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (15.64, (7.12, 24.15)), (0., (0., 0.)), (48.57, (36.85, 60.29))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (41.41, (39.93, 42.88)), (0., (0., 0.)), (54.54, (53.11, 55.97))]
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (15.64, (7.12, 24.15)), (0., (0., 0.)), (35.68, (24.45, 46.92))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (39.75, (38.29, 41.22)), (0., (0., 0.)), (51.68, (50.34, 53.01))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (33.78, (26.07, 41.48)), (0., (0., 0.)), (44.13, (36.05, 52.22))]
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (17.07, (8.24, 25.89)), (0., (0., 0.)), (38.55, (27.13, 49.96))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (44.34, (42.85, 45.83)), (0., (0., 0.)), (52.35, (50.97, 53.73))],
                "Brown_Hair:Wearing_Hat": [(0., (0., 0.)), (21.9, (12.94, 30.86)), (0., (0., 0.)), (45.11, (34.33, 55.89))]
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (14.2, (6.02, 22.39)), (0., (0., 0.)), (41.41, (29.86, 52.96))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (40.47, (39.01, 41.94)), (0., (0., 0.)), (51.95, (50.62, 53.29))],
                "Gray_Hair:Young": [(0., (0., 0.)), (30.22, (19.89, 40.56)), (0., (0., 0.)), (35.5, (24.73, 46.27))]
            }
        ],
    4:
        [
            {
                "Big_Lips:Chubby":
                    [(71.21, (70.85, 71.58)), (17.61, (17.06, 18.16)), (66.92, (66.54, 67.29)), (53.63, (52.91, 54.35))],
                "Bushy_Eyebrows:Blond_Hair":
                    [(92.7, (92.49, 92.91)), (15.64, (7.12, 24.15)), (89.45, (89.2, 89.69)), (39.98, (28.49, 51.47))],
                "Wearing_Lipstick:Male":
                    [(93.93, (93.74, 94.13)), (31.7, (24.13, 39.28)), (93.69, (93.5, 93.89)), (39.99, (32.01, 47.97))],
                "Gray_Hair:Young":
                    [(98.21, (98.1, 98.31)), (28.9, (18.7, 39.11)), (97.55, (97.43, 97.67)), (42.09, (30.98, 53.2))],
            }, {
                "Big_Lips:Chubby": [(0., (0., 0.)), (19.59, (19.02, 20.17)), (0., (0., 0.)), (52.1, (51.38, 52.82))],
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (17.07, (8.24, 25.89)), (0., (0., 0.)), (44.27, (32.62, 55.92))],
                "Gray_Hair:Young": [(0., (0., 0.)), (27.58, (17.53, 37.64)), (0., (0., 0.)), (34.18, (23.5, 44.85))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (37.73, (36.28, 39.18)), (0., (0., 0.)), (50.23, (48.9, 51.56))],
            }, {
                "Big_Lips:Chubby": [(0., (0., 0.)), (18.24, (17.68, 18.79)), (0., (0., 0.)), (54.06, (53.34, 54.78))],
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (15.64, (7.12, 24.15)), (0., (0., 0.)), (35.68, (24.45, 46.92))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (39.01, (37.55, 40.47)), (0., (0., 0.)), (52.44, (51.06, 53.82))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (30.32, (22.84, 37.81)), (0., (0., 0.)), (46.2, (38.08, 54.32))],
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (15.64, (7.12, 24.15)), (0., (0., 0.)), (32.82, (21.81, 43.83))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (36.42, (34.98, 37.86)), (0., (0., 0.)), (49.9, (48.56, 51.23))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (31.01, (23.48, 38.55)), (0., (0., 0.)), (37.92, (30.02, 45.82))],
                "Gray_Hair:Young": [(0., (0., 0.)), (31.54, (21.08, 42.0)), (0., (0., 0.)), (34.18, (23.5, 44.85))],
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (15.64, (7.12, 24.15)), (0., (0., 0.)), (50.0, (38.27, 61.73))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (38.73, (37.27, 40.19)), (0., (0., 0.)), (52.15, (50.77, 53.53))],
                "Brown_Hair:Wearing_Hat": [(0., (0., 0.)), (19.45, (10.88, 28.03)), (0., (0., 0.)), (34.12, (23.84, 44.39))],
                "Big_Lips:Chubby": [(0., (0., 0.)), (19.56, (18.98, 20.13)), (0., (0., 0.)), (58.26, (57.65, 58.87))],
            }, {
                "Bushy_Eyebrows:Blond_Hair": [(0., (0., 0.)), (14.2, (6.02, 22.39)), (0., (0., 0.)), (37.11, (25.78, 48.44))],
                "High_Cheekbones:Smiling": [(0., (0., 0.)), (37.61, (36.16, 39.06)), (0., (0., 0.)), (51.3, (49.99, 52.75))],
                "Gray_Hair:Young": [(0., (0., 0.)), (31.54, (21.08, 42.0)), (0., (0., 0.)), (42.09, (30.98, 53.2))],
                "Brown_Hair:Wearing_Hat": [(0., (0., 0.)), (20.68, (11.9, 29.45)), (0., (0., 0.)), (48.78, (37.95, 59.61))],
            }
        ],
    # 5:
    #     {
    #         "Big_Lips:Chubby":
    #             [(71.8, (71.43, 72.16)), (20.88, (20.29, 21.47)), (67.15, (66.77, 67.52)), (53.51, (52.79, 54.23))],
    #         "Bushy_Eyebrows:Blond_Hair":
    #             [(92.75, (92.54, 92.95)), (12.77, (4.94, 20.6)), (89.57, (89.33, 89.82)), (44.27, (32.62, 55.92))],
    #         "Wearing_Lipstick:Male":
    #             [(93.94, (93.75, 94.13)), (31.7, (24.13, 39.28)), (93.32, (93.12, 93.52)), (35.16, (27.38, 42.93))],
    #         "Gray_Hair:Young":
    #             [(98.14, (98.03, 98.25)), (26.27, (16.36, 36.17)), (97.68, (97.56, 97.8)), (30.22, (19.89, 40.56))],
    #         "High_Cheekbones:Smiling":
    #             [(87.83, (87.57, 88.1)), (37.7, (36.25, 39.15)), (84.4, (84.11, 84.69)), (51.21, (49.88, 52.55))],
    #     },
    # 6:
    #     {
    #         "Big_Lips:Chubby":
    #             [(70.73, (70.37, 71.1)), (14.14, (13.64, 14.64)), (67.43, (67.06, 67.81)), (53.91, (53.19, 54.63))],
    #         "Bushy_Eyebrows:Blond_Hair":
    #             [(92.63, (92.43, 92.84)), (14.2, (6.02, 22.39)), (88.55, (88.29, 88.8)), (41.41, (29.86, 52.96))],
    #         "Wearing_Lipstick:Male":
    #             [(93.6, (93.4, 93.79)), (34.47, (26.73, 42.21)), (93.44, (93.24, 93.64)), (43.44, (35.37, 51.51))],
    #         "Gray_Hair:Young":
    #             [(98.17, (98.06, 98.28)), (27.58, (17.53, 37.64)), (97.16, (97.03, 97.29)), (44.73, (33.54, 55.92))],
    #         "High_Cheekbones:Smiling":
    #             [(87.69, (87.42, 87.95)), (39.03, (37.57, 40.49)), (85.57, (85.29, 85.85)), (52.27, (50.94, 53.6))],
    #         "Brown_Hair:Wearing_Hat":
    #             [(88.57, (88.31, 88.82)), (17.01, (8.87, 25.15)), (85.62, (85.34, 85.9)), (47.56, (36.74, 58.38))],
    #     },
}

mtl_erm_nondisjoint_ablate = {
    2:
        [
            {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (35.21, (32.63, 37.8)), (0., (0., 0.)), (57.01, (54.33, 59.69))],
                "Big_Nose:Male": [(0., (0., 0.)), (35.45, (33.95, 36.96)), (0., (0., 0.)), (53.46, (51.9, 55.03))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (40.25, (36.13, 44.38)), (0., (0., 0.)), (54.41, (50.23, 58.6))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (32.36, (28.45, 36.27)), (0., (0., 0.)), (56.55, (52.4, 60.69))]
            }
        ],
    3:
        [
            {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (35.14, (32.55, 37.72)), (0., (0., 0.)), (53.43, (50.73, 56.13))],
                "Big_Nose:Male": [(0., (0., 0.)), (32.81, (31.34, 34.28)), (0., (0., 0.)), (58.62, (57.07, 60.17))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (35.27, (31.27, 39.26)), (0., (0., 0.)), (48.73, (44.55, 52.9))]
            }, {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (34.6, (32.03, 37.18)), (0., (0., 0.)), (55.26, (52.57, 57.95))],
                "Big_Nose:Male": [(0., (0., 0.)), (35.94, (34.43, 37.45)), (0., (0., 0.)), (59.99, (58.87, 61.11))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (29.63, (22.2, 37.07)), (0., (0., 0.)), (42.06, (34.02, 50.1))]
            }, {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (35.52, (32.93, 38.11)), (0., (0., 0.)), (57.47, (54.8, 60.15))],
                "Big_Nose:Male": [(0., (0., 0.)), (34.27, (32.78, 35.76)), (0., (0., 0.)), (58.7, (57.15, 60.24))],
                "Attractive:Male": [(0., (0., 0.)), (66.57, (65.54, 67.6)), (0., (0., 0.)), (69.25, (68.25, 70.26))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (43.56, (39.4, 47.73)), (0., (0., 0.)), (64.34, (60.32, 68.37))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (34.9, (30.92, 38.89)), (0., (0., 0.)), (52.36, (48.19, 56.54))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (35.85, (28.04, 43.66)), (0., (0., 0.)), (42.06, (34.02, 50.1))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (44.12, (39.94, 48.29)), (0., (0., 0.)), (55.15, (50.97, 59.33))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (34.72, (30.74, 38.7)), (0., (0., 0.)), (50.36, (46.18, 54.54))],
                "Big_Nose:Male": [(0., (0., 0.)), (33.3, (31.82, 34.78)), (0., (0., 0.)), (58.36, (56.82, 59.91))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (42.83, (38.67, 46.99)), (0., (0., 0.)), (58.83, (54.69, 62.96))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (36.0, (31.98, 40.01)), (0., (0., 0.)), (53.82, (49.65, 57.99))],
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (35.9, (33.3, 38.49)), (0., (0., 0.)), (59.45, (56.8, 62.11))]
            }
        ],
    4:
        [
            {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (32.54, (30.01, 35.08)), (0., (0., 0.)), (54.5, (51.8, 57.19))],
                "Big_Nose:Male": [(0., (0., 0.)), (34.63, (33.14, 36.13)), (0., (0., 0.)), (56.7, (55.14, 58.25))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (31.45, (27.57, 35.33)), (0., (0., 0.)), (59.28, (55.17, 63.38))],
                "Blond_Hair:Male": [(0., (0., 0.)), (42.09, (37.94, 46.24)), (0., (0., 0.)), (60.11, (56.0, 64.23))]
            }, {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (35.14, (32.55, 37.72)), (0., (0., 0.)), (55.56, (52.88, 58.25))],
                "Big_Nose:Male": [(0., (0., 0.)), (37.2, (35.68, 38.72)), (0., (0., 0.)), (57.18, (55.63, 58.74))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (31.01, (23.48, 38.55)), (0., (0., 0.)), (43.44, (35.37, 51.51))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (28.54, (24.76, 32.31)), (0., (0., 0.)), (48.55, (44.37, 52.72))]
            }, {
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (40.4, (37.74, 43.05)), (0., (0., 0.)), (60.37, (59.29, 61.45))],
                "Big_Nose:Male": [(0., (0., 0.)), (35.45, (33.95, 36.96)), (0., (0., 0.)), (56.67, (55.11, 58.23))],
                "Attractive:Male": [(0., (0., 0.)), (67.96, (66.94, 68.97)), (0., (0., 0.)), (69.84, (68.96, 70.72))],
                "Blond_Hair:Male": [(0., (0., 0.)), (44.12, (39.94, 48.29)), (0., (0., 0.)), (55.7, (51.53, 59.88))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (42.46, (38.31, 46.62)), (0., (0., 0.)), (59.01, (54.88, 63.14))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (30.36, (26.51, 34.2)), (0., (0., 0.)), (59.82, (55.72, 63.92))],
                "Wearing_Lipstick:Male": [(0., (0., 0.)), (37.92, (30.02, 45.82)), (0., (0., 0.)), (52.42, (44.28, 60.55))],
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (38.26, (35.63, 40.89)), (0., (0., 0.)), (53.66, (50.96, 56.36))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (41.17, (37.04, 45.31)), (0., (0., 0.)), (59.93, (55.81, 64.05))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (31.63, (27.74, 35.52)), (0., (0., 0.)), (53.82, (49.65, 57.99))],
                "Big_Nose:Male": [(0., (0., 0.)), (37.04, (35.53, 38.56)), (0., (0., 0.)), (58.8, (57.25, 60.34))],
                "Attractive:Male": [(0., (0., 0.)), (66.51, (65.48, 67.54)), (0., (0., 0.)), (70.78, (69.79, 71.78))]
            }, {
                "Blond_Hair:Male": [(0., (0., 0.)), (43.38, (39.22, 47.55)), (0., (0., 0.)), (58.09, (53.94, 62.24))],
                "Wearing_Earrings:Male": [(0., (0., 0.)), (40.18, (36.08, 44.28)), (0., (0., 0.)), (57.82, (53.69, 61.95))],
                "Arched_Eyebrows:Male": [(0., (0., 0.)), (39.33, (36.68, 41.97)), (0., (0., 0.)), (57.47, (54.8, 60.15))],
                "Big_Nose:Male": [(0., (0., 0.)), (34.15, (32.66, 35.63)), (0., (0., 0.)), (54.23, (52.67, 55.8))]
            }
        ]
}

STL_ENTRIES = {**stl_erm_disjoint, **stl_erm_nondisjoint, **stl_erm_full_task_ablation}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./outputs/iclr_submission", help="Output directory of ablation plots")
    return parser.parse_args()


def get_n_mtl_task_ablation_average_performances(n, average_type: str):
    if average_type == "disjoint":
        mtl_entries = mtl_erm_disjoint_ablate[n]
    elif average_type == "nondisjoint":
        mtl_entries = mtl_erm_nondisjoint_ablate[n]
    elif average_type == "all":
        mtl_entries = mtl_erm_disjoint_ablate[n] + mtl_erm_nondisjoint_ablate[n]

    mtl_from_avg_accs, mtl_from_wg_accs = [], []
    stl_from_avg_accs, stl_from_wg_accs = [], []
    for entry in mtl_entries:
        for task, (_, wg_from_avg, _, wg_from_wg) in entry.items():
            # TODO currently throwing away CI
            wg_from_avg, _ = wg_from_avg
            wg_from_wg, _ = wg_from_wg

            # Grab MTL values
            mtl_from_avg_accs.append(wg_from_avg)
            mtl_from_wg_accs.append(wg_from_wg)

            # We want to grab the STL values for the exact task
            _, wg_from_avg, _, wg_from_wg = STL_ENTRIES[task]
            # TODO currently throwing away CI
            wg_from_avg, _ = wg_from_avg
            wg_from_wg, _ = wg_from_wg

            stl_from_avg_accs.append(wg_from_avg)
            stl_from_wg_accs.append(wg_from_wg)

    mtl_from_avg_accs = sum(mtl_from_avg_accs) / len(mtl_from_avg_accs)
    mtl_from_wg_accs = sum(mtl_from_wg_accs) / len(mtl_from_wg_accs)
    stl_from_avg_accs = sum(stl_from_avg_accs) / len(stl_from_avg_accs)
    stl_from_wg_accs = sum(stl_from_wg_accs) / len(stl_from_wg_accs)
    return mtl_from_avg_accs, mtl_from_wg_accs, stl_from_avg_accs, stl_from_wg_accs


def get_mtl_task_ablation_average_performances(average_type: str, save_name: str):
    """Computes the individual average task performances from MTL and STL"""
    assert average_type in ["disjoint", "nondisjoint", "all"]
    x = list(range(2, 5))

    mtl_from_avg_accs = {}
    mtl_from_wg_accs = {}
    stl_from_avg_accs = {}
    stl_from_wg_accs = {}

    for n in x:
        mtl_from_avg_accs[n], mtl_from_wg_accs[n], stl_from_avg_accs[n], stl_from_wg_accs[n] = \
            get_n_mtl_task_ablation_average_performances(n, average_type)

    data = pd.DataFrame(
        {
            "# Tasks": x,
            "MTL: Mean of WG Acc (Avg Ckpt)": [mtl_from_avg_accs[k] for k in x],
            "MTL: Mean of WG Acc (WG Ckpt": [mtl_from_wg_accs[k] for k in x],
            "STL: Mean of WG Acc (Avg Ckpt)": [stl_from_avg_accs[k] for k in x],
            "STL: Mean of WG Acc (WG Ckpt)": [stl_from_wg_accs[k] for k in x],
        }
    )

    plt.clf()
    plt.cla()
    ax = sns.lineplot(
        x="# Tasks",
        y="Accuracy",
        hue="hue",
        data=pd.melt(
            data,
            id_vars=["# Tasks"],
            value_vars=[
                "MTL: Mean of WG Acc (Avg Ckpt)",
                "MTL: Mean of WG Acc (WG Ckpt",
                "STL: Mean of WG Acc (Avg Ckpt)",
                "STL: Mean of WG Acc (WG Ckpt)",
            ],
            value_name="Accuracy",
            var_name="hue"
        )
    )
    ax.set_title(f"MTL Task Ablation: {average_type.upper()}")
    ax.set_xticks(x)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_name)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("Creating MTL Disjoint Ablation plot")
    get_mtl_task_ablation_average_performances(
        average_type="disjoint",
        save_name=os.path.join(args.out_dir, "mtl_disjoint_ablate.png"),
    )
    logger.info("Creating MTL Non-Disjoint Ablation plot")
    get_mtl_task_ablation_average_performances(
        average_type="nondisjoint",
        save_name=os.path.join(args.out_dir, "mtl_nondisjoint_ablate.png"),
    )
    logger.info("Creating MTL Disjoint + Non-Disjoint Ablation plot")
    get_mtl_task_ablation_average_performances(
        average_type="all",
        save_name=os.path.join(args.out_dir, "mtl_all_ablate.png"),
    )


if __name__ == "__main__":
    main()
