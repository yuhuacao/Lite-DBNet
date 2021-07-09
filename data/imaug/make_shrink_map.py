import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

__all__ = ['MakeShrinkMap']


class MakeShrinkMap(object):

    def __init__(self, min_text_size=8, shrink_ratio=0.4, **kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):
        image = data['image']
        text_polys = data['polys']  # 文本区域多边形
        ignore_tags = data['ignore_tags']  # 是否需要忽略该文本区域

        h, w = image.shape[:2]
        # 检验文本区域，去除面积过小的
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            # 获取文本区域的宽和高
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # 如果是需要忽略的文本区域，创建mask
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # shrink操作
                shrinked = self.shrink_polygon_pyclipper(polygon)
                if shrinked.size == 0:
                    # 如果是需要忽略的文本区域，创建mask
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons: (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            # 将所有点坐标控制在图像宽高范围内
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                # 对点进行倒序
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def shrink_polygon_pyclipper(self, polygon):
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked = padding.Execute(-distance)  # 向内收缩
        if not shrinked:
            shrinked = np.array(shrinked)
        else:
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
        return shrinked

    def polygon_area(self, polygon):
        """
        compute polygon area
        """
        area = 0
        q = polygon[-1]
        for p in polygon:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2.0
        # return cv2.contourArea(polygon)
