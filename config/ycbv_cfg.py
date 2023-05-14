dataset_name = "ycbv"
diameters ={
            1:172.063,
            2:269.573, 
            3:198.377,
            4:120.543, 
            5:196.463,
            6:89.797,  
            7:142.543, 
            8:114.053, 
            9:129.540, 
            10:197.796,
            11:259.534, 
            12:259.566, 
            13:161.922, 
            14:124.990, 
            15:226.170,
            16:237.299, 
            17:203.973, 
            18:121.365, 
            19:174.746, 
            20:217.094,
            21:102.903
            }
#OUTPUT_DIR = "output/geo/ycbv/a6_cPnP_AugAAETrunc_BG0.5_Rsym_ycbv_real_pbr_visib20_10e"
# INPUT = dict(
#     DZI_PAD_SCALE=1.5,
#     TRUNCATE_FG=True,
#     CHANGE_BG_PROB=0.5,
#     COLOR_AUG_PROB=0.8,
#     COLOR_AUG_TYPE="code",
#     COLOR_AUG_CODE=(
#         "Sequential(["
#         # Sometimes(0.5, PerspectiveTransform(0.05)),
#         # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
#         # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
#         "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
#         "Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),"
#         "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
#         "Sometimes(0.3, Invert(0.2, per_channel=True)),"
#         "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
#         "Sometimes(0.5, Multiply((0.6, 1.4))),"
#         "Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))"
#         "], random_order = False)"
#         # aae
#     ),
# )

SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=30,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CHECKPOINT_PERIOD=1,
)

DATASETS = dict(
    DATA_ROOT_DIR = 'datasets/ycbv/ycbv',
    BATCH_SIZE = 64,
    TRAIN = ("train_real","train_pbr",),#("train_real","train_synt"),#, "train_pbr", "train_synt"),
    #TRAIN=("train_real", "train_pbr"),
    #TRAIN=("train_real", "train_pbr"),
    TEST = ("test",),
    OBJ_IDS = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21),
    SELECTED_OBJ_ID = 1,
    IMG_SIZE = [480,640],
    DZI_SCALE_RATIO = 0.25,
    DZI_SHIFT_RATIO = 0.25,
    DZI_PAD_RATIO = 1.5,
    OBJS = {
    1:"002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2:"003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3:"004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4:"005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5:"006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6:"007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7:"008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8:"009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9:"010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10:"011_banana",  # [-18.6730, 12.1915, -1.4635]
    11:"019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12:"021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13:"024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14:"025_mug",  # [-8.4675, -0.6995, -1.6145]
    15:"035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16:"036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17:"037_scissors",  # [7.0535, -28.1320, 0.0420]
    18:"040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19:"051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20:"052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21:"061_foam_brick", # [-0.0805, 0.0805, -8.2435]
    },
    FILTER_INVALID=True,
    MODEL_PT_NUM = 4096,
    NUM_SAMPLE_POINTS = 4096,
    NN_DIST_TH = 0.05,
    INPUT_SIZE = 256,
    MODEL_D = diameters,

    SYM_OBJS=["024_bowl", "052_extra_large_clamp", "061_foam_brick"],  # ycbv  "036_wood_block", "051_large_clamp", 
)

class ConfigRandLA:
    
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = DATASETS["NUM_SAMPLE_POINTS"]  # Number of input points
    in_c = 9
    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]
config_rand = ConfigRandLA()

DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=8,
    FILTER_VISIB_THR=0.2,
    TRAIN_BATCH_SIZE = 8,
    VAL_BATCH_SIZE = 128,
)

MODEL = dict(
    n_mesh_node = (DATASETS["MODEL_PT_NUM"]),
    feat_dim = (128),
    checkpoints = 'train_log/ycb/checkpoints',
    model_pth = 'datasets/ycbv/ycbv/kps',
    ffb_config = config_rand,
    resnet_dir = 'models/cnn/ResNet_pretrained_mdl',
    model_d = diameters,
    neighbor_dis_th = 0.06,
    model_name = 'ycbv'
    )

MESH = dict(
    MESH_DIR = ('datasets/ycbv/ycbv/kps')
)
VAL = dict(
    DATASET_NAME="ycbvposecnn",
    SPLIT_TYPE="",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="ycbv_test_targets_keyframe.json",
    ERROR_TYPES="AUCadd,AUCadi,AUCad,ad,ABSadd,ABSadi,ABSad",
    USE_BOP=True,  # whether to use bop toolkit
    EVAL_CACHED = True,
    EVAL_PRINT_ONLY = False,

)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
