import unittest

from annotation_validator import (
    AnnotationValidator,
    NormalizedBBox,
    UnifiedAnnotation,
    calculate_polygon_area,
    compute_raster_polygon_iou,
    compute_shape_score,
)


class TestAnnotationValidator(unittest.TestCase):
    def setUp(self):
        self.validator = AnnotationValidator()

    def test_shoelace_formula_triangle_and_trapezoid(self):
        """Перевірка точного математичного розрахунку площ геометрії за формулою Гаусса."""
        # Прямокутний трикутник з катетами 30 і 40 (площа: 0.5 * 30 * 40 = 600)
        triangle = [(0.0, 0.0), (30.0, 0.0), (0.0, 40.0)]
        self.assertAlmostEqual(calculate_polygon_area(triangle), 600.0, places=5)

        # Трапеція: основи a=10, b=20, висота h=10 (площа: 0.5 * (10 + 20) * 10 = 150)
        trapezoid = [(5.0, 0.0), (15.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        self.assertAlmostEqual(calculate_polygon_area(trapezoid), 150.0, places=5)

    def test_both_path_raster_iou(self):
        """Перевірка розрахунку перекриття через растрову маску, коли path є в обох об'єктів."""
        # Два однакових квадрати 100x100
        poly1 = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        poly2 = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        ref_box = NormalizedBBox(0.0, 0.0, 100.0, 100.0)

        iou = compute_raster_polygon_iou(poly1, poly2, ref_box, grid_size=32)
        self.assertAlmostEqual(iou, 1.0, places=2)

        # Неперетинні квадрати
        poly3 = [(200.0, 200.0), (300.0, 200.0), (300.0, 300.0), (200.0, 300.0)]
        ref_box_disjoint = NormalizedBBox(0.0, 0.0, 300.0, 300.0)
        iou_disjoint = compute_raster_polygon_iou(poly1, poly3, ref_box_disjoint, grid_size=32)
        self.assertEqual(iou_disjoint, 0.0)

    def test_polygon_vs_bbox_containment_and_shape_score(self):
        """Тест оцінки якості форми при порівнянні багатокутника та BBox."""
        bbox = NormalizedBBox(0.0, 0.0, 100.0, 100.0)
        gt_obj = UnifiedAnnotation(id="1", obj_type="barrier", shape_type="bBox", bbox=bbox)

        # Полігон, що акуратно вписаний всередину (трикутник площею 5000 у коробці 10000)

        inside_poly = [(0.0, 0.0), (100.0, 0.0), (50.0, 100.0)]
        pred_obj_inside = UnifiedAnnotation(
            id="a", obj_type="barrier", shape_type="poly", bbox=bbox, path=inside_poly
        )
        score_inside = compute_shape_score(gt_obj, pred_obj_inside)
        self.assertAlmostEqual(score_inside, 0.75, places=2)

        # Полігон, де точки грубо виходять за межі коробки
        outside_poly = [(0.0, 0.0), (200.0, 0.0), (50.0, 200.0)]
        pred_obj_outside = UnifiedAnnotation(
            id="b", obj_type="barrier", shape_type="poly", bbox=bbox, path=outside_poly
        )
        score_outside = compute_shape_score(gt_obj, pred_obj_outside)
        self.assertLess(score_outside, score_inside)

    def test_real_dataset_with_barrier_path(self):
        """Перевірка обробки реального датасету зі збереженням та оцінкою контуру path."""
        gt_data = [
            {"objects": [{"nm": "0.7", "type": "road_barrier", "shape": "poly"}]},
            {"objects": [{"nm": "0.7", "x1": 596.49, "y1": 415.58, "x2": 876.28, "y2": 503.45}]}
        ]
        pred_data = [
            {"objects": [{"nm": "0.8", "type": "road_barrier", "shape": "poly"}]},
            {
                "objects": [
                    {
                        "nm": "0.8",
                        "x1": 594.88,
                        "y1": 414.32,
                        "x2": 876.33,
                        "y2": 510.44,
                        "path": [
                            [
                                [876.33, 488.99], [818.55, 484.32], [789.15, 480.11],
                                [749.95, 471.51], [725.55, 461.32], [691.75, 444.32],
                                [666.95, 429.91], [636.75, 419.32], [597.35, 414.32],
                                [594.88, 421.71], [600.72, 422.33], [663.60, 438.44],
                                [762.15, 493.71], [874.88, 504.62]
                            ]
                        ]
                    }
                ]
            }
        ]

        report = self.validator.evaluate(gt_data, pred_data)
        self.assertEqual(len(report.matched_pairs), 1)

        pair = report.matched_pairs[0]
        # BBox перекриття дуже високе (~0.92)
        self.assertGreater(pair.bbox_iou, 0.9)
        # Оцінка форми враховує вписування складного path у габарити еталона
        self.assertGreater(pair.shape_score, 0.5)
        self.assertLessEqual(pair.shape_score, 1.0)

    def test_polygon_vs_bbox_span_shrinkage(self):
        """Перевірка, що недотягнуті габарити або крихітна фігура всередині BBox не отримують бонус."""
        bbox = NormalizedBBox(0.0, 0.0, 100.0, 100.0)
        gt_obj = UnifiedAnnotation(id="1", obj_type="car", shape_type="bBox", bbox=bbox)

        # 1. Повнорозмірний трикутник [0..100] по X та [0..100] по Y
        full_poly = [(0.0, 0.0), (100.0, 0.0), (50.0, 100.0)]
        pred_full = UnifiedAnnotation(id="a", obj_type="car", shape_type="poly", bbox=bbox, path=full_poly)
        score_full = compute_shape_score(gt_obj, pred_full)

        # 2. Полігон, який втратив половину висоти (span_y = 0.5)
        half_height_poly = [(0.0, 0.0), (100.0, 0.0), (50.0, 50.0)]
        pred_half = UnifiedAnnotation(id="b", obj_type="car", shape_type="poly", bbox=bbox, path=half_height_poly)
        score_half = compute_shape_score(gt_obj, pred_half)

        # 3. Крихітний квадрат 10x10 всередині коробки 100x100
        tiny_poly = [(45.0, 45.0), (55.0, 45.0), (55.0, 55.0), (45.0, 55.0)]
        pred_tiny = UnifiedAnnotation(id="c", obj_type="car", shape_type="poly", bbox=bbox, path=tiny_poly)
        score_tiny = compute_shape_score(gt_obj, pred_tiny)

        # Оцінка повинна строго зменшуватися
        self.assertGreater(score_full, score_half)
        self.assertGreater(score_half, score_tiny)
        # Крихітний полігон повинен отримати оцінку, близьку до нуля
        self.assertLess(score_tiny, 0.05)

    def test_zero_matches_and_empty_frames(self):
        """Перевірка крайових випадків: пропущений кадр (0 збігів) та True Negative (порожній кадр)."""
        # 1. Еталон має об'єкти, а розмітник здав порожній файл -> бал 0.0
        gt_data = [
            {"objects": [{"nm": "1", "type": "car", "shape": "bBox"}]},
            {"objects": [{"nm": "1", "x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}]}
        ]
        pred_empty = [{"objects": []}, {"objects": []}]
        report_missed = self.validator.evaluate(gt_data, pred_empty)
        self.assertEqual(report_missed.overall_score, 0.0)
        self.assertEqual(report_missed.mean_shape_score, 0.0)
        self.assertEqual(report_missed.mean_attribute_accuracy, 0.0)

        # 2. Обидва файли порожні (True Negative) -> бал 1.0 (100%)
        report_both_empty = self.validator.evaluate(pred_empty, pred_empty)
        self.assertEqual(report_both_empty.overall_score, 1.0)


if __name__ == "__main__":
    unittest.main()