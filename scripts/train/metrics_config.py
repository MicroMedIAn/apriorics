from torchmetrics import Accuracy, JaccardIndex, Precision, Specificity

from apriorics.metrics import (
    DetectionSegmentationMetrics,
    DiceScore,
    Recall,
    SegmentationAUC,
)

METRICS = {
    "all": [
        JaccardIndex(task="binary", ignore_index=0),
        DiceScore(),
        Accuracy(task="binary"),
        Precision(task="binary"),
        Recall(task="binary"),
        Specificity(task="binary"),
        SegmentationAUC(),
    ],
    "PHH3": [DetectionSegmentationMetrics(flood_fill=True)],
}
