NUM_WORDS = 36
NUM_KERNELS = [32, 16]
KERNEL_SIZES = [4,3]
STRIDES = [3,1]
NUM_DENSE_LAYER_UNITS = [50, 30]
ATTENTION_UNITS = [30, 25]

NUM_USERS = 614505
NUM_ITEMS = 132006
USER_EMBEDDING_DIM = 16
ITEM_EMBEDDING_DIM = 16

BATCH_SIZE = 1000

TRAIN_DATA_FILE = "preprocessed_train_data.txt" # "./data/preprocessed_train_data.txt"
TEST_DATA_FILE = "preprocessed_test_data.txt" # "./data/preprocessed_test_data.txt"
PRODUCT_IMAGE_DIR = "product_images" # "./data/product_images"
