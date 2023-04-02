"""Contains macros needed across all experiments/scripts."""

ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

CXR_ATTRIBUTES = ["Infiltration", "Fibrosis", "Hernia", "Mass", "Pleural_Thickening", "No Finding", "Pneumonia", "Atelectasis", "Nodule", "Edema", "Consolidation", "Cardiomegaly", "Effusion", "Emphysema", "Pneumothorax", "Male", "Old"]

# Dictionary of task lists based on the type of ablation experiments we want to run
TASKS = {

    ##########################
    # COMPLETE MTL TASK SETS #
    ##########################

    # 5 tasks to tune STL methods on before running spurious ID on them
    "SPURIOUS_ID_TUNE":
        [
            "Attractive:Eyeglasses",
            "Smiling:High_Cheekbones",
            "Pointy_Nose:Rosy_Cheeks",
            "Oval_Face:Rosy_Cheeks",
            "Young:Attractive",
        ],
    "SPURIOUS_ID_ALL": ATTRIBUTES,
    "SPURIOUS_ID_ALL_CXR": CXR_ATTRIBUTES,

    # 2 sets of ablations over disjoint tasks, for each: 1x MTL(2), 3x MTL(3), 3x MTL(4)
    "MTL_ABLATE_DISJOINT":
        [
            # PREVIOUS RUNS

            # MTL(2)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling"],

            # MTL(3)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Gray_Hair:Young"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Gray_Hair:Young"],

            # MTL(4)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "High_Cheekbones:Smiling"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Gray_Hair:Young", "High_Cheekbones:Smiling"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male", "Gray_Hair:Young"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat", "Big_Lips:Chubby"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Gray_Hair:Young", "Brown_Hair:Wearing_Hat"],
        ],

    # 2 sets of ablations over nondisjoint tasks, for each: 1x MTL(2), 3x MTL(3), 3x MTL(4)
    "MTL_ABLATE_NONDISJOINT":
        [
            # MTL(2)
            ["Arched_Eyebrows:Male", "Big_Nose:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male"],

            # MTL(3)
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Earrings:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Lipstick:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Attractive:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Wearing_Lipstick:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Big_Nose:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Arched_Eyebrows:Male"],

            # MTL(4)
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Earrings:Male", "Blond_Hair:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Lipstick:Male", "Wearing_Earrings:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Attractive:Male", "Blond_Hair:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Wearing_Lipstick:Male", "Arched_Eyebrows:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Big_Nose:Male", "Attractive:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Arched_Eyebrows:Male", "Big_Nose:Male"],
        ],

    # 3 pairs of pairwise disjoint tasks
    "MTL_STL_COMPARISON":
        [
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"], ["Wearing_Lipstick:Male", "Gray_Hair:Young"],
            ["High_Cheekbones:Smiling", "Wearing_Lipstick:Male"]
        ],

    # 5 pairs of pairwise disjoint tasks
    "MTL_DISJOINT":
        [
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"],
            ["Wearing_Lipstick:Male", "Gray_Hair:Young"],
            ["High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat"],
            ["No_Beard:Wearing_Lipstick", "Young:Chubby"],
            ["Bangs:Wearing_Hat", "Pointy_Nose:Heavy_Makeup"],
        ],

    # 5 pairs of pairwise nondisjoint tasks
    "MTL_NONDISJOINT":
        [
            ["Wearing_Earrings:Male", "Attractive:Male"],
            ["No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup"],
            ["Attractive:Gray_Hair", "Big_Nose:Gray_Hair"],
            ["Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick"],
            ["Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"],
        ],

    ############################
    # INCOMPLETE MTL TASK SETS #
    ############################
    # exclude repeats from above

    # 4 NEW pairs of semantically similar task pairs
    "MTL_SIMILAR":
        [
            ["Big_Nose:Wearing_Lipstick", "High_Cheekbones:Smiling"],
            ["Big_Lips:Goatee", "Wearing_Lipstick:Male"],
            ["Bags_Under_Eyes:Double_Chin", "High_Cheekbones:Rosy_Cheeks"],
            ["Blond_Hair:Wearing_Hat", "Brown_Hair:Wearing_Hat"],
        ],

    # 2 NEW pairs of strongly spuriously correlated task pairs
    "MTL_STRONG": [
        ["Wearing_Lipstick:Male", "High_Cheekbones:Smiling"],
        ["Heavy_Makeup:Male", "Wearing_Earrings:Male"],
    ],

    # 2 NEW pairs of weakly spuriously correlated task pairs
    "MTL_WEAK": [
        ["Big_Lips:Chubby", "Young:Chubby"],
        ["High_Cheekbones:Rosy_Cheeks", "Brown_Hair:Wearing_Hat"],
    ],
}

# Defines param grids for tuning methods
GRIDS = {
    "erm": {
        "WD": [1e-4, 1e-3, 1e-2, 1e-1],
        "LR": [1e-5, 5e-5, 1e-4],
        "BATCH_SIZE": [128],
    },
    "suby": {
        "WD": [1e-2, 1e-1, 1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64],
    },
    "mtl_erm": {
        "WD": [1e-4, 1e-3, 1e-2, 1e-1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64, 128],
    },
    "mtl_suby": {
        "WD": [1e-2, 1e-1, 1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64],
    },
}

# Defines params for established methods
PARAMS = {
    # STL baseline methods
    "erm": {
        "WD": 1e-4,
        "LR": 1e-4,
        "BATCH_SIZE": 128,
        "EPOCHS": 50,
    },
    "suby": {
        "WD": 1e-2,
        "LR": 1e-3,
        "BATCH_SIZE": 128,
        "EPOCHS": 60,
    },
    "rwy": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "EPOCHS": 60,
    },
    "jtt":
        {
            "WD": 1e-1,
            "LR": 1e-5,
            "BATCH_SIZE": 128,
            "EPOCHS": 50,
            "T": 1,
            "LAM_UP": 50,  # some extras here for jtt, handled with an if statement
        },

    # MTL ERM methods, tuned with different task weightings and checkpointing
    "mtl_erm_avg_ckpt_static_equal_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_avg_ckpt_static_delta_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_avg_ckpt_dynamic_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_static_equal_mtl_weighting": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_static_delta_mtl_weighting": {
        "WD": 1e-3,
        "LR": 1e-3,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_dynamic_mtl_weighting": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },

    # THESE AREN'T TUNED FOR MTL
    "mtl_suby": {
        "WD": 1e-2,
        "LR": 1e-3,
        "BATCH_SIZE": 128,
        "EPOCHS": 60,
    },
    "mtl_rwy": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "EPOCHS": 60,
    },
    "mtl_jtt":
        {
            "WD": 1e-1,
            "LR": 1e-5,
            "BATCH_SIZE": 128,
            "EPOCHS": 50,
            "T": 1,
            "LAM_UP": 50,  # some extras here for jtt, handled with an if statement
        },

    # CLIP models
    # TODO(vliu but not necessary) find a way to decouple this dict for --opt and --model
    "clip_erm": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 128,  # BATCH_SIZE wasn't tuned for CLIP tuning
        "EPOCHS": 50
    }
}
