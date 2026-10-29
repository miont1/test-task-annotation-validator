import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NormalizedBBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class UnifiedAnnotation:
    id: str
    obj_type: str
    shape_type: str
    bbox: NormalizedBBox
    attributes: dict[str, Any] = field(default_factory=dict)
    path: list[tuple[float, float]] | None = None


@dataclass
class MatchedPair:
    gt_id: str
    pred_id: str
    obj_type: str
    bbox_iou: float
    shape_score: float
    attribute_accuracy: float
    attribute_details: dict[str, bool]


@dataclass
class ValidationReport:
    overall_score: float
    f1_score: float
    precision: float
    recall: float
    mean_bbox_iou: float
    mean_shape_score: float
    mean_attribute_accuracy: float
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    matched_pairs: list[MatchedPair]
    missed_ground_truth: list[dict[str, str]]
    false_positives: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Серіалізує звіт у стандартний словник Python."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Повертає звіт у форматі валідного JSON-рядка."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def normalize_coordinates(x1: float, y1: float, x2: float, y2: float) -> NormalizedBBox:
    """Нормалізує координати для запобігання інверсії осей."""
    return NormalizedBBox(
        x1=min(x1, x2),
        y1=min(y1, y2),
        x2=max(x1, x2),
        y2=max(y1, y2)
    )


def compute_bbox_iou(box1: NormalizedBBox, box2: NormalizedBBox) -> float:
    """Аналітичний розрахунок IoU для прямокутників із захистом від ділення на нуль."""
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    if intersection_area <= 0.0:
        return 0.0

    union_area = box1.area + box2.area - intersection_area
    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def calculate_polygon_area(points: list[tuple[float, float]]) -> float:
    """Обчислює площу довільного простого багатокутника за формулою Гаусса (Shoelace formula)."""
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def rasterize_polygon(
    points: list[tuple[float, float]],
    grid_w: int,
    grid_h: int,
    ref_box: NormalizedBBox
) -> bytearray:
    """Раструє полігон у бінарну маску через метод трасування променя (Even-odd rule)."""
    mask = bytearray(grid_w * grid_h)
    bw = max(ref_box.width, 1e-5)
    bh = max(ref_box.height, 1e-5)

    norm_pts = [
        (
            ((x - ref_box.x1) / bw) * grid_w,
            ((y - ref_box.y1) / bh) * grid_h
        )
        for x, y in points
    ]
    n = len(norm_pts)
    if n < 3:
        return mask

    for py in range(grid_h):
        sample_y = py + 0.5
        node_x = []
        for i in range(n):
            p1 = norm_pts[i]
            p2 = norm_pts[(i + 1) % n]
            if (p1[1] < sample_y <= p2[1]) or (p2[1] < sample_y <= p1[1]):
                x_intersect = p1[0] + (sample_y - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0])
                node_x.append(x_intersect)

        node_x.sort()
        for k in range(0, len(node_x) - 1, 2):
            start_x = max(0, int(math.ceil(node_x[k])))
            end_x = min(grid_w, int(math.ceil(node_x[k + 1])))
            row_offset = py * grid_w
            for px in range(start_x, end_x):
                mask[row_offset + px] = 1

    return mask


def compute_raster_polygon_iou(
    poly1: list[tuple[float, float]],
    poly2: list[tuple[float, float]],
    ref_box: NormalizedBBox,
    grid_size: int = 64
) -> float:
    """Обчислює IoU двох полігонів через бінарні растрові маски фіксованого розміру."""
    mask1 = rasterize_polygon(poly1, grid_size, grid_size, ref_box)
    mask2 = rasterize_polygon(poly2, grid_size, grid_size, ref_box)

    inter_count = 0
    union_count = 0
    total_pixels = grid_size * grid_size

    for i in range(total_pixels):
        v1 = mask1[i]
        v2 = mask2[i]
        if v1 and v2:
            inter_count += 1
        if v1 or v2:
            union_count += 1

    if union_count == 0:
        return 0.0

    return inter_count / union_count


def compute_shape_score(obj_gt: UnifiedAnnotation, obj_pred: UnifiedAnnotation) -> float:
    """
    Розраховує якість форми з двостороннім контролем меж (Boundary Span)
    та мультиплікативним масштабуванням для пари Polygon vs BBox.
    """
    has_gt_path = obj_gt.path is not None and len(obj_gt.path) >= 3
    has_pred_path = obj_pred.path is not None and len(obj_pred.path) >= 3

    if not has_gt_path and not has_pred_path:
        return 1.0

    union_box = NormalizedBBox(
        x1=min(obj_gt.bbox.x1, obj_pred.bbox.x1),
        y1=min(obj_gt.bbox.y1, obj_pred.bbox.y1),
        x2=max(obj_gt.bbox.x2, obj_pred.bbox.x2),
        y2=max(obj_gt.bbox.y2, obj_pred.bbox.y2)
    )

    if has_gt_path and has_pred_path:
        return compute_raster_polygon_iou(obj_gt.path, obj_pred.path, union_box, grid_size=64)

    # Випадок Polygon vs BBox
    poly = obj_pred.path if has_pred_path else obj_gt.path
    bbox = obj_gt.bbox if has_pred_path else obj_pred.bbox

    if bbox.area <= 0.0 or not poly:
        return 0.0

    # 1. Потрапляння точок контуру в допустимі межі прямокутника
    tolerance = 2.0
    inside_count = sum(
        1 for x, y in poly
        if (bbox.x1 - tolerance <= x <= bbox.x2 + tolerance) and
           (bbox.y1 - tolerance <= y <= bbox.y2 + tolerance)
    )
    containment_ratio = inside_count / len(poly)

    # 2. Двосторонній контроль меж (Boundary Span): екстремуми полігону проти габаритів BBox
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    poly_min_x, poly_max_x = min(xs), max(xs)
    poly_min_y, poly_max_y = min(ys), max(ys)

    span_x = max(0.0, min(poly_max_x, bbox.x2) - max(poly_min_x, bbox.x1)) / max(bbox.width, 1e-5)
    span_y = max(0.0, min(poly_max_y, bbox.y2) - max(poly_min_y, bbox.y1)) / max(bbox.height, 1e-5)
    extent_factor = min(1.0, span_x) * min(1.0, span_y)

    # 3. Відношення площі контуру до площі BBox
    poly_area = calculate_polygon_area(poly)
    fill_ratio = min(poly_area / bbox.area, 1.0)

    # 4. Мультиплікативне масштабування:
    # Базова форма = 50% потрапляння точок + 50% щільність заповнення.
    # Підсумковий бал масштабується на extent_factor (повноту охоплення габаритів).
    base_accuracy = (containment_ratio * 0.5) + (fill_ratio * 0.5)
    score = base_accuracy * extent_factor

    return max(0.0, min(1.0, score))


def parse_two_block_format(payload: list[dict[str, Any]]) -> list[UnifiedAnnotation]:
    """Парсить двоблочну структуру JSON зі збереженням точок контуру path."""
    if len(payload) < 2:
        return []

    meta_objects = {
        obj.get("nm"): obj
        for obj in payload[0].get("objects", [])
        if "nm" in obj
    }

    unified: list[UnifiedAnnotation] = []
    for coord_obj in payload[1].get("objects", []):
        nm = coord_obj.get("nm")
        if not nm:
            continue

        meta = meta_objects.get(nm, {})
        obj_type = meta.get("type", "unknown")
        shape_type = meta.get("shape", "bBox")

        raw_x1 = coord_obj.get("x1")
        raw_y1 = coord_obj.get("y1")
        raw_x2 = coord_obj.get("x2")
        raw_y2 = coord_obj.get("y2")

        if any(v is None for v in (raw_x1, raw_y1, raw_x2, raw_y2)):
            continue

        bbox = normalize_coordinates(float(raw_x1), float(raw_y1), float(raw_x2), float(raw_y2))

        combined_attrs = {}
        if isinstance(meta.get("attributes"), dict):
            combined_attrs.update(meta["attributes"])
        if isinstance(coord_obj.get("attributes"), dict):
            combined_attrs.update(coord_obj["attributes"])

        raw_path = coord_obj.get("path")
        parsed_path: list[tuple[float, float]] | None = None
        if raw_path and isinstance(raw_path, list) and len(raw_path) > 0:
            contour = raw_path[0]
            if isinstance(contour, list):
                parsed_path = [
                    (float(pt[0]), float(pt[1]))
                    for pt in contour
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2
                ]

        unified.append(
            UnifiedAnnotation(
                id=str(nm),
                obj_type=obj_type,
                shape_type=shape_type,
                bbox=bbox,
                attributes=combined_attrs,
                path=parsed_path
            )
        )
    return unified


def evaluate_attributes(gt_attrs: dict[str, Any], pred_attrs: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    """Порівнює значущі бізнес-атрибути розмітки."""
    keys = set(gt_attrs.keys()).union(set(pred_attrs.keys()))
    if not keys:
        return 1.0, {}

    details = {}
    matches = 0
    for key in keys:
        is_match = (gt_attrs.get(key) == pred_attrs.get(key))
        details[key] = is_match
        if is_match:
            matches += 1

    return matches / len(keys), details


class AnnotationValidator:
    def evaluate(self, gt_raw: list[dict[str, Any]], pred_raw: list[dict[str, Any]]) -> ValidationReport:
        gt_items = parse_two_block_format(gt_raw)
        pred_items = parse_two_block_format(pred_raw)

        classes = set([item.obj_type for item in gt_items] + [item.obj_type for item in pred_items])

        matched_pairs: list[MatchedPair] = []
        missed_gt: list[dict[str, str]] = []
        false_positives: list[dict[str, str]] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for c in classes:
            class_gt = [item for item in gt_items if item.obj_type == c]
            class_pred = [item for item in pred_items if item.obj_type == c]

            if not class_gt:
                for p in class_pred:
                    false_positives.append({"id": p.id, "type": c})
                total_fp += len(class_pred)
                continue

            if not class_pred:
                for g in class_gt:
                    missed_gt.append({"id": g.id, "type": c})
                total_fn += len(class_gt)
                continue

            candidates: list[tuple[float, int, int]] = []
            for g_idx, g in enumerate(class_gt):
                for p_idx, p in enumerate(class_pred):
                    iou = compute_bbox_iou(g.bbox, p.bbox)
                    if iou > 0.0:
                        candidates.append((iou, g_idx, p_idx))

            candidates.sort(key=lambda x: x[0], reverse=True)

            matched_gt_set = set()
            matched_pred_set = set()

            for iou, g_idx, p_idx in candidates:
                if g_idx in matched_gt_set or p_idx in matched_pred_set:
                    continue

                matched_gt_set.add(g_idx)
                matched_pred_set.add(p_idx)
                total_tp += 1

                gt_obj = class_gt[g_idx]
                pred_obj = class_pred[p_idx]

                shape_score = compute_shape_score(gt_obj, pred_obj)
                attr_acc, attr_details = evaluate_attributes(gt_obj.attributes, pred_obj.attributes)

                matched_pairs.append(
                    MatchedPair(
                        gt_id=gt_obj.id,
                        pred_id=pred_obj.id,
                        obj_type=c,
                        bbox_iou=round(iou, 4),
                        shape_score=round(shape_score, 4),
                        attribute_accuracy=round(attr_acc, 4),
                        attribute_details=attr_details
                    )
                )

            for g_idx, g in enumerate(class_gt):
                if g_idx not in matched_gt_set:
                    missed_gt.append({"id": g.id, "type": c})
                    total_fn += 1

            for p_idx, p in enumerate(class_pred):
                if p_idx not in matched_pred_set:
                    false_positives.append({"id": p.id, "type": c})
                    total_fp += 1

        if not gt_items and not pred_items:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            mean_iou = 1.0
            mean_shape = 1.0
            mean_attr_acc = 1.0
            overall_score = 1.0
        else:
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            mean_iou = sum(p.bbox_iou for p in matched_pairs) / len(matched_pairs) if matched_pairs else 0.0
            mean_shape = sum(p.shape_score for p in matched_pairs) / len(matched_pairs) if matched_pairs else 0.0
            mean_attr_acc = sum(p.attribute_accuracy for p in matched_pairs) / len(matched_pairs) if matched_pairs else 0.0

            overall_score = round(0.30 * f1 + 0.30 * mean_iou + 0.25 * mean_shape + 0.15 * mean_attr_acc, 4)

        return ValidationReport(
            overall_score=overall_score,
            f1_score=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            mean_bbox_iou=round(mean_iou, 4),
            mean_shape_score=round(mean_shape, 4),
            mean_attribute_accuracy=round(mean_attr_acc, 4),
            true_positive_count=total_tp,
            false_positive_count=total_fp,
            false_negative_count=total_fn,
            matched_pairs=matched_pairs,
            missed_ground_truth=missed_gt,
            false_positives=false_positives
        )


if __name__ == "__main__":

    script_dir = os.path.dirname(__file__)
    ground_truth_file = os.path.join(script_dir, r"ground_truth/ground_truth.json")
    annotated_file = os.path.join(script_dir, r"annotated/human_annotation.json")

    with open(ground_truth_file, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    with open(annotated_file, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    validator = AnnotationValidator()
    report = validator.evaluate(gt_raw, pred_raw)

    print("=" * 70)
    print("           ПІДСУМКОВИЙ ЗВІТ ВАЛІДАЦІЇ РОЗМІТКИ КАДРУ           ")
    print("=" * 70)
    summary = report.to_dict()
    print(f"Загальний бал (Overall Score):    {summary['overall_score'] * 100:.2f}%")
    print(f"F1-міра (F1 Score):               {summary['f1_score'] * 100:.2f}%")
    print(f"Точність виявлення (Precision):   {summary['precision'] * 100:.2f}%")
    print(f"Повнота покриття (Recall):        {summary['recall'] * 100:.2f}%")
    print(f"Середній BBox IoU:                {summary['mean_bbox_iou'] * 100:.2f}%")
    print(f"Середня якість форми (Shape):     {summary['mean_shape_score'] * 100:.2f}%")
    print(f"Збіг атрибутів:                   {summary['mean_attribute_accuracy'] * 100:.2f}%")
    print("-" * 70)
    print(f"Статистика об'єктів: TP={report.true_positive_count} | FP={report.false_positive_count} | FN={report.false_negative_count}")
    print("-" * 70)
    print("Зіставлені пари фігур:")
    for idx, p in enumerate(report.matched_pairs, 1):
        attr_flag = "OK" if p.attribute_accuracy == 1.0 else "FAIL"
        print(
            f"  {idx}. [{p.obj_type}] GT ID={p.gt_id:<4} <-> Pred ID={p.pred_id:<4} | "
            f"BBox IoU: {p.bbox_iou * 100:>5.1f}% | Форма: {p.shape_score * 100:>5.1f}% | "
            f"Атрибути ({attr_flag}): {p.attribute_details}"
        )

    if report.missed_ground_truth:
        print("\nПропущені еталонні фігури (False Negatives):")
        for missed in report.missed_ground_truth:
            print(f"  - ID {missed['id']} (тип: {missed['type']})")

    if report.false_positives:
        print("\nЗайві розмітки людини (False Positives):")
        for fp in report.false_positives:
            print(f"  - ID {fp['id']} (тип: {fp['type']})")
    print("=" * 70)