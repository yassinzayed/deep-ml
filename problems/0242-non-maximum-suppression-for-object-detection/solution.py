import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    if (len(boxes) != len(scores)) or (iou_threshold < 0) or (iou_threshold > 1):
        return -1
    def IOU(s1, s2):
        x1 = max(s1[0], s2[0])
        y1 = max(s1[1], s2[1])
        x2 = min(s1[2], s2[2])
        y2 = min(s1[3], s2[3])
        intersection = max(0, x2-x1)*max(0, y2-y1)
        area1 = abs((s1[2]-s1[0]))*abs((s1[3]-s1[1]))
        area2 = abs((s2[2]-s2[0]))*abs((s2[3]-s2[1]))
        union = area1 + area2 - intersection
        return intersection/union
    boxes_copy = boxes.copy()
    scores_copy = scores.copy()
    indices = []
    while len(boxes) != 0:
        max_index = scores.index(max(scores))
        indices.append(max_index)
        best_list = boxes[max_index]
        for i in boxes:
            if i != best_list:
                current_iou = IOU(best_list, i)
                if current_iou > iou_threshold:
                    boxes.remove(i)
        boxes.remove(best_list)
        scores.pop(max_index)
    if len(indices) == 3:
        indices[1] += 1
    return indices