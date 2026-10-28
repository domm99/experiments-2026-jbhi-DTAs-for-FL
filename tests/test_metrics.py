import math
import unittest

from tests import _path  # noqa: F401
import torch

from utils import CLASS_TO_INDEX, classification_metrics_from_confusion_matrix, update_confusion_matrix


def metrics_for(y_true: list[str], y_pred: list[str]) -> dict:
    confusion_matrix = torch.zeros((3, 3), dtype=torch.long)
    update_confusion_matrix(
        confusion_matrix,
        torch.tensor([CLASS_TO_INDEX[y] for y in y_true]),
        torch.tensor([CLASS_TO_INDEX[y] for y in y_pred]),
    )
    return classification_metrics_from_confusion_matrix(confusion_matrix)


class MacroMetricsTest(unittest.TestCase):

    def test_perfect_classifier_with_missing_class_scores_one(self):
        labels = ['normal', 'normal', 'hyper', 'normal', 'hyper']
        metrics = metrics_for(labels, labels)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        self.assertTrue(math.isnan(metrics['recall_hypo']))
        self.assertTrue(math.isnan(metrics['precision_hypo']))

    def test_matches_scikit_learn_convention(self):
        # sklearn: precision_score(y, p, average='macro', zero_division=0)
        y_true = ['normal', 'normal', 'hyper', 'hyper']
        y_pred = ['normal', 'hypo', 'hyper', 'normal']
        metrics = metrics_for(y_true, y_pred)
        # classes present: hypo (pred only), normal, hyper
        self.assertAlmostEqual(metrics['precision'], (0.0 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(metrics['recall'], (0.0 + 0.5 + 0.5) / 3)
        self.assertEqual(metrics['cm_normal_hypo'], 1)
        self.assertEqual(metrics['cm_hyper_normal'], 1)

    def test_empty_confusion_matrix_is_nan(self):
        metrics = classification_metrics_from_confusion_matrix(torch.zeros((3, 3), dtype=torch.long))
        self.assertEqual(metrics['prediction_count'], 0)
        self.assertTrue(math.isnan(metrics['accuracy']))


if __name__ == '__main__':
    unittest.main()
