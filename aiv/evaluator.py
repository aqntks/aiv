
import numpy as np
from typing import List, Tuple, Dict, Any


class OCREvaluator:
    
    def __init__(self, preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]) -> None:
        """
        OCREvaluator 클래스의 초기화 함수.

        Args:
            preds (List[Dict[str, Any]]): 예측 결과 리스트.
            gts (List[Dict[str, Any]]): Ground Truth 리스트.
        """

        self.preds = preds
        self.gts = gts

        self.pred_texts, self.pred_boxes = self.jsons_to_ocr(self.preds)
        self.gt_texts, self.gt_boxes = self.jsons_to_ocr(self.gts)

    def jsons_to_ocr(self, jsons: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]]]:
        """
        OCR 데이터에서 텍스트와 박스 정보를 추출하는 함수.

        Args:
            jsons (List[Dict[str, Any]]): OCR 데이터가 포함된 JSON 리스트.

        Returns:
            Tuple[List[str], List[List[float]]]: 텍스트 리스트와 박스 좌표 리스트.
        """

        texts, boxes = [], []

        for json in jsons:
            text, box = "", [0.0, 0.0, 0.0, 0.0]

            if "ocr" in json:
                text = json["ocr"]
            if "box" in json:
                box = json["box"]
                box = [float(p) for p in box]
            
            texts.append(text)
            boxes.append(box)

        return texts, boxes

    def calculate_text_accuracy(self) -> float:
        """
        GT 텍스트 리스트와 Pred 텍스트 리스트 간의 정확도 계산 함수.
        두 리스트는 같은 길이를 가져야 합니다.

        Returns:
            float: 텍스트 일치율로 계산된 정확도.
        """
        
        gt_texts, pred_texts = self.gt_texts, self.pred_texts

        assert len(gt_texts) == len(pred_texts)
        
        correct_count = 0
        total_count = len(gt_texts)
        
        for gt_text, pred_text in zip(gt_texts, pred_texts):
            if gt_text.replace(" ", "") == pred_text.replace(" ", ""):
                correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        return accuracy

    def calculate_iou(self, gt_box: List[float], pred_box: List[float]) -> float:
        """
        IOU(Intersection over Union)를 계산하는 함수.
        
        Args:
            gt_box (List[float]): Ground Truth 박스 좌표 [x, y, w, h].
            pred_box (List[float]): 예측 박스 좌표 [x, y, w, h].

        Returns:
            float: GT와 Pred 박스 간의 IOU 값.
        """

        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]

        inter_x1 = max(gt_x1, pred_x1)
        inter_y1 = max(gt_y1, pred_y1)
        inter_x2 = min(gt_x2, pred_x2)
        inter_y2 = min(gt_y2, pred_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        intersection_area = inter_w * inter_h

        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        union_area = gt_area + pred_area - intersection_area

        if union_area == 0:
            return 0
        iou = intersection_area / union_area

        return iou

    def map_iou(self, gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> np.ndarray:
        """
        GT와 Pred 박스들 간의 IOU 맵을 생성하는 함수.

        Args:
            gt_boxes (List[List[float]]): Ground Truth 박스 리스트.
            pred_boxes (List[List[float]]): 예측 박스 리스트.

        Returns:
            np.ndarray: IOU 매트릭스 (GT 박스 vs Pred 박스).
        """

        iou_map = np.zeros((len(gt_boxes), len(pred_boxes)))

        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_map[i, j] = self.calculate_iou(gt_box, pred_box)
        
        return iou_map

    def calculate_box_precision_and_recall(self, iou_threshold: float = 0.5) -> Tuple[float, float]:
        """
        특정 IOU 임계값에서 Precision(정밀도)과 Recall(재현율)을 계산하는 함수.

        Args:
            iou_threshold (float, optional): IOU 임계값. 기본값은 0.5.

        Returns:
            Tuple[float, float]: 정밀도(Precision)와 재현율(Recall).
        """

        gt_boxes, pred_boxes = self.gt_boxes, self.pred_boxes

        iou_map = self.map_iou(gt_boxes, pred_boxes)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou_values = iou_map[gt_idx, :]
            max_iou_idx = np.argmax(iou_values)
            max_iou = iou_values[max_iou_idx]

            if max_iou >= iou_threshold:
                true_positives += 1
            else:
                false_negatives += 1

        false_positives = len(pred_boxes) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        return precision, recall
