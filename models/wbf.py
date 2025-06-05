from ensemble_boxes import weighted_boxes_fusion
from config import IMAGE_SIZE, WBF_IOU_THR, WBF_SKIP_BOX_THR, WBF_WEIGHTS

def run_wbf(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0, weights=None, image_size=(704, 520)):
    """
    boxes_list: List of list of boxes for each model, each box: [x1, y1, x2, y2] (absolute coords)
    scores_list: List of list of scores for each model
    labels_list: List of list of labels for each model
    iou_thr: IoU threshold for WBF
    skip_box_thr: Skip boxes with score lower than this
    weights: List of weights for each model
    image_size: (width, height) of the image
    Returns:
        boxes: fused boxes in absolute coords [x1, y1, x2, y2]
        scores: fused scores
        labels: fused labels
    """
    w, h = image_size
    # Convert boxes to normalized format [0, 1]
    norm_boxes_list = []
    for boxes in boxes_list:
        norm_boxes = []
        for box in boxes:
            norm_boxes.append([box[0]/w, box[1]/h, box[2]/w, box[3]/h])
        norm_boxes_list.append(norm_boxes)

    boxes, scores, labels = weighted_boxes_fusion(
        norm_boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    # Convert boxes back to absolute coords
    abs_boxes = []
    for box in boxes:
        abs_boxes.append([box[0]*w, box[1]*h, box[2]*w, box[3]*h])

    return abs_boxes, scores, labels