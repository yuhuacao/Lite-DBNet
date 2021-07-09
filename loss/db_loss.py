import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class BalanceLoss(nn.Module):
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction='none')

    def forward(self, pred, gt, mask=None):
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)[0]
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


def ohem_single(score, gt_text, training_mask, ohem_ratio):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (
        int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(
            1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * ohem_ratio, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(
            1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    # 将负样本得分从高到低排序
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]
    # 选出 得分高的 负样本 和正样本 的 mask
    selected_mask = ((score >= threshold) |
                     (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(
        1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks, ohem_ratio):
    scores = scores.numpy()
    gt_texts = gt_texts.numpy()
    training_masks = training_masks.numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[
                i, :, :], ohem_ratio))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.to_tensor(selected_masks)

    return selected_masks


class DBLoss(nn.Module):

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts['maps']
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map, label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map, label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
        losses = {'loss': loss_all, \
                  "loss_shrink_maps": loss_shrink_maps, \
                  "loss_threshold_maps": loss_threshold_maps, \
                  "loss_binary_maps": loss_binary_maps}
        return losses
