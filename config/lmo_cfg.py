


OUTPUT_DIR = "output/geo/ycbv/a6_cPnP_AugAAETrunc_BG0.5_Rsym_ycbv_real_pbr_visib20_10e"

diameters = {
    
            1:102.099,  # 1
            2:247.506,
            3:167.355,
            4:172.492,
            5:201.404,  # 5
            6:154.546,  # 6
            7:124.264,
            8:261.472,  # 8
            9:108.999,  # 9
            10:164.628,  # 10
            11:175.889,  # 11
            12:145.543,  # 12
            13:278.078,
            14:282.601,
            15:212.358,
}
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))"
        "], random_order = False)"
        # aae
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=128,
    TOTAL_EPOCHS=50,
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
dataset_name = "lmo"
DATASETS = dict(
    DATA_ROOT_DIR = 'datasets/lm/linemod/',
    #BATCH_SIZE = 16,
    TRAIN = ("train_pbr",),#, "train_pbr", "train_synt"),"real","fuse","renders"
    #TRAIN=("train_real", "train_pbr"),
    #TRAIN=("train_real", "train_pbr"),
    TEST = ("test",),
    OBJ_IDS = (1,5,6,8,9,10,11,12),
    SELECTED_OBJ_ID = 1,
    IMG_SIZE = [480,640],
    DZI_SCALE_RATIO = 0.25,
    DZI_SHIFT_RATIO = 0.25,
    DZI_PAD_RATIO = 1.5,
    OBJS = {
    1: "ape",
    #2: 'benchvise',
    #3: 'bowl',
    #4: 'camera',
    5: "can",
    6: "cat",
    #7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    #13: 'iron',
    #14: 'lamp',
    #15: 'phone'
    },
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/ycbv/test/test_bboxes/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json",
    ),
    FILTER_INVALID=True,
    MODEL_PT_NUM = 4096,
    NUM_SAMPLE_POINTS = 4096,
    NN_DIST_TH = 0.05,
    INPUT_SIZE = 256,
    MODEL_D = diameters,


    SYM_OBJS=["eggbox",],  # ycbv
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
    NUM_WORKERS=4,
    FILTER_VISIB_THR=0.2,
    TRAIN_BATCH_SIZE = 24,
    VAL_BATCH_SIZE = 128,
)

MODEL = dict(
    n_mesh_node = (DATASETS["MODEL_PT_NUM"]),
    feat_dim = (128),
    checkpoints = 'train_log/lm/checkpoints',
    model_pth = 'datasets/lm/linemod/kps',
    ffb_config = config_rand,
    resnet_dir = 'models/cnn/ResNet_pretrained_mdl',
    model_d = diameters,
    neighbor_dis_th = 0.02,
    model_name = 'lmo'
    )

MESH = dict(
    MESH_DIR = ('datasets/lm/linemod/kps')
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


