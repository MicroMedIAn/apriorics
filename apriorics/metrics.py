from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from pathaia.util.basic import ifnone
from skimage.measure import label
from torchmetrics import CatMetric, Metric, MetricCollection, Recall
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod

from apriorics.masks import flood_mask


@torch.jit.unused
def _forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Iteratively call forward for each metric.

    Positional arguments (args) will be passed to every metric in the collection, while
    keyword arguments (kwargs) will be filtered based on the signature of the individual
    metric.
    """
    res = {}
    for k, m in self.items():
        out = m(*args, **m._filter_kwargs(**kwargs))
        if isinstance(out, dict):
            res.update(out)
        else:
            res[k] = out
    return res


def _compute(self) -> Dict[str, Any]:
    res = {}
    for k, m in self.items():
        out = m.compute()
        if isinstance(out, dict):
            res.update(out)
        else:
            res[k] = out
    return res


MetricCollection.forward = _forward
MetricCollection.compute = _compute


def _recall_compute(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    average: str,
    mdmc_average: Optional[str],
) -> torch.Tensor:
    """Computes precision from the stat scores: true positives, false positives, true
    negatives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class
            inputs (on top of the ``average`` parameter)
    """
    numerator = tp
    denominator = tp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = ((tp | fn | fp) == 0).nonzero().cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != AverageMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=1,
    )


def recall_compute(self) -> torch.Tensor:
    """Computes the recall score based on inputs passed in to ``update`` previously.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element
            tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C``
            stands  for the number of classes
    """
    tp, fp, _, fn = self._get_final_stats()
    return _recall_compute(tp, fp, fn, self.average, self.mdmc_reduce)


Recall.compute = recall_compute


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""
    Optionally reduces input tensor by either averaging or summing its values.

    Args:
        x: input tensor.
        reduction: reduction method, either "mean", "sum" or "none".

    Returns:
        Reduced version of x.
    """
    if reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        return x


def _flatten(x: torch.Tensor) -> torch.Tensor:
    r"""
    Flattens input tensor but keeps first dimension.

    Args:
        x: input tensor of shape (N, ...).

    Returns:
        Flattened version of `x` of shape (N, .).
    """
    return x.view(x.shape[0], -1)


def dice_score(
    input: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    Computes dice score (given by :math:`D(p, t) = \frac{2|pt|+s}{|p|+|t|+s}`) between
    predicted input tensor and target ground truth.

    Args:
        input: predicted input tensor of shape (N, ...).
        target: target ground truth tensor of shape (N, ...).
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".

    Returns:
             Computed dice score, optionally reduced using specified reduction method.
    """
    target = _flatten(target).to(dtype=input.dtype)
    input = _flatten(input)
    inter = (target * input).sum(-1)
    sum = target.sum(-1) + input.sum(-1)
    dice = (2 * inter + smooth) / (sum + smooth)
    return _reduce(dice, reduction=reduction)


class DiceScore(CatMetric):
    r"""
    `torch.nn.Module` for dice loss (given by
    :math:`D(p, t) = \frac{2|pt|+s}{|p|+|t|+s}`) computation.

    Args:
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".
    """

    def __init__(self, smooth: float = 1, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.reduction = reduction

    def update(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Optional[torch.Tensor]:
        dice = dice_score(input, target, smooth=self.smooth, reduction=self.reduction)
        return super().update(dice)

    def compute(self) -> torch.Tensor:
        dices = super().compute()
        dices = torch.as_tensor(dices)
        return _reduce(dices, reduction=self.reduction)


class SegmentationAUC(Metric):
    def __init__(self, clf_thresholds: Optional[Sequence[float]] = None, **kwargs):
        super().__init__(**kwargs)
        if clf_thresholds is None:
            clf_thresholds = torch.arange(0, 1.05, 0.05)
        else:
            clf_thresholds = torch.as_tensor(clf_thresholds)
        self.register_buffer("clf_thresholds", clf_thresholds, persistent=False)

        self.add_state(
            "tp",
            default=torch.zeros(len(self.clf_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros(len(self.clf_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros(len(self.clf_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tn",
            default=torch.zeros(len(self.clf_thresholds)),
            dist_reduce_fx="sum",
        )

    def update(self, input: torch.Tensor, target: torch.Tensor):
        input = input.view(-1)[None]
        target = target.bool().view(-1)[None]
        pred = input >= self.clf_thresholds[:, None]
        tp = pred & target
        fp = pred & ~target
        fn = ~pred & target
        tn = ~pred & ~target
        self.tp += tp.sum(-1)
        self.fp += fp.sum(-1)
        self.fn += fn.sum(-1)
        self.tn += tn.sum(-1)

    def compute(self):
        tpr = self.tp / (self.tp + self.fn + 1e-7)
        fpr = self.fp / (self.tn + self.fp + 1e-7)
        prec = (self.tp + 1e-7) / (self.tp + self.fp + 1e-7)
        res = {"AUROC": -torch.trapz(tpr, fpr), "AUPRC": -torch.trapz(prec, tpr)}
        return res


class DetectionSegmentationMetrics(Metric):
    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        clf_threshold: float = 0.5,
        area_threshold: int = 50,
        flood_fill: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.iou_thresholds = ifnone(
            iou_thresholds, np.arange(0, 1, 0.05).round(2).tolist()
        )
        self.clf_threshold = clf_threshold
        self.area_threshold = area_threshold
        self.flood_fill = flood_fill

        self.add_state(
            "tp_pred",
            default=torch.zeros(len(self.iou_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tp_target",
            default=torch.zeros(len(self.iou_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros(len(self.iou_thresholds)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros(len(self.iou_thresholds)),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ):
        if x is not None and self.flood_fill:
            x = np.ascontiguousarray(
                (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255),
                dtype=np.uint8,
            )
        for k, (y_hat, y) in enumerate(zip(input, target)):
            y_hat = (y_hat > self.clf_threshold).detach().cpu().numpy()
            labels_pred, n_pred = label(y_hat, return_num=True)
            to_substract = 0
            for k_pred in range(n_pred):
                mask_pred = labels_pred == (k_pred + 1)
                if mask_pred.sum() < self.area_threshold:
                    labels_pred[mask_pred] = 0
                    to_substract += 1
                    n_pred -= 1
                else:
                    labels_pred[mask_pred] -= to_substract

            labels_target, n_target = label(y.bool().cpu().numpy(), return_num=True)
            if n_target == 0:
                self.fp += n_pred
                continue
            if n_pred == 0:
                self.fn += n_target
                continue
            missing = torch.ones(
                (len(self.iou_thresholds), n_pred, n_target), device=self.fp.device
            )
            for k_pred in range(n_pred):
                mask_pred = labels_pred == (k_pred + 1)
                if x is not None and self.flood_fill:
                    img = x[k]
                    mask_pred = flood_mask(img, mask_pred, n=20)
                ii, jj = np.nonzero(mask_pred)
                y0, y1 = ii.min(), ii.max()
                x0, x1 = jj.min(), jj.max()
                area_pred = (y1 - y0 + 1) * (x1 - x0 + 1)
                bbox_pred = np.array([x0, y0, x1, y1])
                for k_target in range(n_target):
                    mask_target = labels_target == (k_target + 1)
                    ii, jj = np.nonzero(mask_target)
                    y0, y1 = ii.min(), ii.max()
                    x0, x1 = jj.min(), jj.max()
                    area_target = (y1 - y0 + 1) * (x1 - x0 + 1)
                    bbox_target = np.array([x0, y0, x1, y1])
                    bbox_inter = np.concatenate(
                        (
                            np.maximum(bbox_pred[:2], bbox_target[:2]),
                            np.minimum(bbox_pred[2:], bbox_target[2:]),
                        )
                    )
                    x0, y0, x1, y1 = bbox_inter
                    area_inter = (y1 - y0 + 1) * (x1 - x0 + 1)
                    iou = area_inter / (area_pred + area_target - area_inter + 1e-7)
                    missing[iou > self.iou_thresholds, k_pred, k_target] = 0
            self.fp += missing.all(2).sum(-1)
            self.fn += missing.all(1).sum(-1)
            self.tp_target += (1 - missing).any(1).sum(-1)
            self.tp_pred += (1 - missing).any(2).sum(-1)

    def compute(self) -> Dict[str, torch.Tensor]:
        precisions = self.tp_pred / (self.tp_pred + self.fp + 1e-7)
        recalls = (self.tp_target + 1e-7) / (self.tp_target + self.fn + 1e-7)
        j_10 = self.iou_thresholds.index(0.1)
        j_25 = self.iou_thresholds.index(0.25)
        j_50 = self.iou_thresholds.index(0.5)
        j_75 = self.iou_thresholds.index(0.75)
        res = {
            "precision_10": precisions[j_10],
            "precision_25": precisions[j_25],
            "precision_50": precisions[j_50],
            "precision_75": precisions[j_75],
            "precision_50+": precisions[j_50:].mean(),
            "recall_10": recalls[j_10],
            "recall_25": recalls[j_25],
            "recall_50": recalls[j_50],
            "recall_75": recalls[j_75],
            "recall_50+": recalls[j_50:].mean(),
        }
        return res


class PredictionHist(Metric):
    def __init__(self, density: bool = False, bin_size: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.bins = np.arange(0, 1 + bin_size, bin_size)
        self.add_state(
            "hists", default=torch.zeros((2, int(1 / bin_size))), dist_reduce_fx="sum"
        )
        self.density = density
        self.bin_size = bin_size

    def update(self, input, target):
        input = input.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        hists = np.zeros((2, int(1 / self.bin_size)))
        for i in range(2):
            mask = target == i
            hists[i] += np.histogram(input[mask], self.bins)[0]
        self.hists += torch.as_tensor(hists).to(self.hists)

    def compute(self):
        if self.density:
            return self.hists / (self.hists.sum(-1, keepdim=True) * self.bin_size)
        else:
            return self.hists
