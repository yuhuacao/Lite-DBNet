from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):  # 求并集
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):  # 求IOU
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):  # 求交集
            return Polygon(pD).intersection(Polygon(pG)).area

        matchedSum = 0  # pred和gt匹配的总个数
        numGlobalCareGt = 0  # 整张图关注的gt数量
        numGlobalCareDet = 0  # 整张图关注的预测文本框数量
        detMatched = 0  # 预测文本框匹配个数
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        detPolPoints = []
        gtDontCarePolsNum = []  # 可忽略的gt框的索引列表
        detDontCarePolsNum = []  # 可忽略的预测框的索引列表
        pairs = []  # pred和gt的匹配对
        detMatchedNums = []

        for n in range(len(gt)):  # 遍历gt框
            points = gt[n]['points']  # 获取每个框的点坐标
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']  # 获取是否忽略的标记

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPols.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)    # 在gtDontCarePolsNum列表加入框的gtPols索引

        for n in range(len(pred)):  # 遍历预测框
            detPol = pred[n]['points']  # 获取每个框的点坐标
            if not Polygon(detPol).is_valid or not Polygon(detPol).is_simple:
                continue
            detPols.append(detPol)

            if len(gtDontCarePolsNum) > 0:  # 如果有需要忽略的gt文本框
                for dontCarePol in gtDontCarePolsNum:  # 遍历需要忽略的gt文本框
                    dontCarePol = gtPols[dontCarePol]  # 获取每个gt框的点坐标
                    intersected_area = get_intersection(dontCarePol, detPol)  # 计算当前预测框与gt框的交集面积
                    pdDimensions = Polygon(detPol).area  # 预测框的面积
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if precision > self.area_precision_constraint:  # 重叠比例大于阈值，则加入可忽略的预测框的索引列表
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            # 计算完iouMat后
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):  # 暴力循环每种配对
                    # 如果当前gtNum框和detNum框都没配对， 并且都不是要忽略的
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum \
                            and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:  # 仅通过iou阈值寻找gt和pred匹配的文本框
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1  # 标记
                            detMatched += 1  # pred框匹配个数加1
                            pairs.append({'gt': gtNum, 'det': detNum})  # 匹配对+1
                            detMatchedNums.append(detNum)  # 保存匹配好的pred框的索引

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))   # 计算真正保留下的gt和pred的文本框个数

        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare   # 匹配上的个数/有效gt个数
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare  # 匹配上的个数/有效预测个数
        # f1 score 对precision和recall求调和平均
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0

        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / \
                                                                    (methodRecall + methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)