import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess(object):
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5, **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        # 获取文本框轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]
        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            # 将原来相对光滑的轮廓曲线转为折线
            epsilon = 0.005 * cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            points = approx.reshape(-1, 2)
            if len(points) < 4:
                continue
            # 文字区域概率均值
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            # 轮廓膨胀
            box = self.unclip(points)
            box = box.reshape(-1, 2)

            # 保证文字框最小边长不能太小
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            # box与原图匹配
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        # 获取文本框轮廓
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]
        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            # 保证文字框最小边长不能太小
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            # 文字区域概率均值
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            # 轮廓膨胀
            box = self.unclip(points).reshape(-1, 1, 2)
            # 保证膨胀后文字框最小边长不能太小
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            # box与原图匹配
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        # 求包含contour的最小矩形
        bounding_box = cv2.minAreaRect(contour)
        # 获取各点坐标并排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        # 将四个点从左上角开始顺时针排序
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        # bounding_box[1]返回的是矩形的长和宽，取最小值
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)  # 生成mask，标记为1的
        # 计算bitmap为1的区域的probabity均值作为score
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list, is_output_polygon=False):
        pred = outs_dict['maps'].cpu().detach().numpy()
        pred = pred[:, 0, :, :]   # prob map
        segmentation = pred > self.thresh  # 二值化
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = shape_list[batch_index][:2]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], src_w, src_h)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], src_w, src_h)

            boxes_batch.append({'points': boxes})
            scores_batch.append(scores)
        return boxes_batch
