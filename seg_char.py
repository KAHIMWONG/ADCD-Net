import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as op
import cv2
import numpy as np
from paddleocr import TextDetection
from glob import glob
import pickle
from tqdm import tqdm

class TextDetector:
    def __init__(self):
        self.model = TextDetection(model_name="PP-OCRv5_server_det")

    def get_mask(self, img_path, save_path):
        output = self.model.predict(input=img_path, batch_size=1)

        # Extract detection results (assuming single image input)
        res = output[0]
        polys = res['dt_polys']
        scores = res['dt_scores']  # Optional: can filter based on scores if needed

        # Load image to get dimensions
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Create binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, poly in enumerate(polys):
            if scores[i] > 0.5:  # Optional threshold for confidence
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Save the mask as PNG (multiply by 255 for visibility: white text regions on black background)
        cv2.imwrite(save_path, mask * 255)

if __name__ == '__main__':
    detector = TextDetector()

    pkl_dir = '/data/jesonwong47/DocForgData/path_pkl'
    pkl_list = glob(os.path.join(pkl_dir, '*.pkl'))

    # remove RTM in pkl_list
    pkl_list = [pkl for pkl in pkl_list if 'RTM' not in pkl]

    new_pkl_dir = '/data/jesonwong47/DocForgData/path_pkl_ocr'
    ocr_root = '/data/jesonwong47/DocForgData/ocr_mask'
    for each_pkl in pkl_list:
        ds_name = op.basename(each_pkl).replace('.pkl', '')
        print(f'Processing dataset: {ds_name}')
        with open(each_pkl, 'rb') as f:
            path_list = pickle.load(f)
        new_path_list = []
        ocr_dir = op.join(ocr_root, ds_name)
        os.makedirs(ocr_dir, exist_ok=True)
        for each_path in tqdm(path_list):
            img_path, mask_path = each_path
            img_name = op.basename(mask_path)
            ocr_path = op.join(ocr_dir, img_name)
            detector.get_mask(img_path, ocr_path)
            new_path_list.append((img_path, mask_path, ocr_path))
        with open(op.join(new_pkl_dir, f'{ds_name}.pkl'), 'wb') as f:
            pickle.dump(new_path_list, f)
