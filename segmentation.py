from ultralytics import YOLO, SAM
import cv2
from typing import Union
import numpy as np

DEVICE = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'

class Segmentation:
    def __init__(self, yolo_model_path: str, sam_model_path: str):
        self.yolo = YOLO(yolo_model_path)
        self.sam = SAM(sam_model_path)

    def detect(self, img: Union[str, np.ndarray], conf: float = 0.75):
        # YOLO source accepts both str paths and numpy arrays natively
        results = self.yolo.predict(source=img, conf=conf, iou=0.45, max_det=300, device=DEVICE, save=False)
        return results

    def segment(self, img: Union[str, np.ndarray],yolo_results=None):
        if yolo_results is None:
            bboxes = None
        bboxes = yolo_results[0].boxes.xyxy
        sam_results = self.sam(yolo_results[0].orig_img, bboxes=bboxes, verbose=False, save=False, device=DEVICE)
        return sam_results

    def show_bboxes(self, results, save_path: str = None): 
        img = results[0].orig_img.copy()  # Added .copy() to avoid modifying original in-memory image
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if save_path:
            cv2.imwrite(save_path, img)
        return img
    
    def show_masks(self, sam_results, opacity: float = 0.5, save_path: str = None):
        assert 0 <= opacity <= 1, "Opacity must be between 0 and 1"
        img = sam_results[0].orig_img.copy() 
        mask = sam_results[0].masks.data.cpu().numpy().sum(axis=0)

        # highlight mask in red
        masked_pixels = img[mask > 0].astype(np.float32)
        masked_pixels += [0, 0, 255 * opacity]
        img[mask > 0] = np.clip(masked_pixels, 0, 255).astype(np.uint8)
        # img[mask > 0] = img[mask > 0] + [0, 0, 255*opacity]

        if save_path:
            cv2.imwrite(save_path, img)
        return img


if __name__ == "__main__":

    # Example usage
    yolo_model_path = r"/home/tr23920-a/Webcam_Monitoring/volcanic_vision/models/louis.pt"
    sam_model_path = r"/home/tr23920-a/Webcam_Monitoring/yolo_training/models/sam_b.pt"
    segmentation = Segmentation(yolo_model_path, sam_model_path)

    image_path = r"example.png"
    image = cv2.imread(image_path)

    # yolo
    yolo_res = segmentation.detect(image)
    bbox_img = segmentation.show_bboxes(yolo_res, save_path="yolo_result.png")

    # sam
    sam_res = segmentation.segment(image, yolo_results=yolo_res)
    mask_img = segmentation.show_masks(sam_res, save_path="sam_result.png")
