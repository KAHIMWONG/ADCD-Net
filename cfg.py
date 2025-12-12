import datetime
import os
import os.path as op
import albumentations as A
import cv2

gpus = '0,1,2,3'
mode = 'general_val'  # 'train', 'val', 'general_val'
check_val = False
train_bs = 20
val_bs = 4

step_per_epoch = 700
ds_len = cnt_per_epoch = step_per_epoch * train_bs
print_log_step = 100
val_epoch = 25
init_S = 0
epochs = 150
# ------------------ MODEL CFG -------------------

ce_w = 3
rec_w = 1
focal_w = [0.001, 0.005, 0.02, 0.1]
norm_w = 0.1

# ======= Set evaluation distortion here =======
multi_jpeg_val = True  # able to use multi jpeg distortion
jpeg_record = False  # manually set multi jpeg distortion record
min_qf = 75  # minimum jpeg quality factor
shift_1p = False  # shift 1 pixel for evaluation

# A.Downscale(scale_min=0.95, scale_max=0.95, p=1)
# A.Resize(height=int(512*0.98), width=int(512*0.98), interpolation=cv2.INTER_LINEAR)
# A.CropNonEmptyMaskIfExists(height=256, width=256, p=1.0),
# A.Downscale(scale_min=0.7, scale_max=0.7)
# A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.7, 0.7), p=1.0)
val_aug = None # A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.7, 0.7), p=1.0)  # other distortions can be added here
# ======= Set evaluation distortion here =======

# ------------------ MODEL CFG -------------------

root = '/data/jesonwong47/DocTamper'
ckpt = '/data/jesonwong47/DocTamper/exp_out/ADCDNet/Log_v11240350/ckpt/Ep_150_0.7741.pth'
docres_ckpt_path = '/data/jesonwong47/DocTamper/ckpt/docres.pkl'

test_set = ['T-SROIE', 'TPIC-13', 'OSTF', 'RTM']  # ['TestingSet', 'FCD', 'SCD']
test_set_sample = [_name + '_sample' for _name in test_set]
val_name_list = test_set_sample if mode == 'train' else test_set
val_sample_n = 100

# -------------------- FIX ----------------------

data_root = op.join(root, 'DocTamperV1')
ocr_root = op.join(root, 'DocTamperData')
qt_path = op.join(root, 'exp_data/qt_table.pk')
jpeg_record_dir = op.join(root, 'exp_data/pks')

lr = 3e-4
min_lr = 1e-5
weight_decay = 1e-4
min_qf = 75
img_size = 256
total_step = step_per_epoch * epochs

exp_root_name = 'ADCDNet'
now_time = datetime.datetime.now()
now_time = 'Log_v%02d%02d%02d%02d/' % (now_time.month, now_time.day, now_time.hour, now_time.minute)
exp_dir = op.join(root, f'exp_out/{exp_root_name}', now_time)
tb_log = op.join(exp_dir, 'tb_log')
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(tb_log, exist_ok=True)

