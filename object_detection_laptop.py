import argparse

import cv2
import numpy as np


DISPLAY_SCALE = 1.0
HEADER_FONT_SCALE = 1.0
TEXT_THICKNESS = 3
BOX_THICKNESS = 4

CENTER_TOLERANCE_PIXELS = 50
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIN_RADIUS_PIXELS = 16
MAX_RADIUS_PIXELS = 110
OPTIMAL_MIN_RADIUS = 34
OPTIMAL_MAX_RADIUS = 58
MAX_DISPLAYED_DETECTIONS = 3
TRACKER_SMOOTHING = 0.35
TRACKER_MAX_MISSED_FRAMES = 6

TOO_CLOSE_COLOR = (40, 40, 255)
OPTIMAL_COLOR = (80, 255, 80)
TOO_FAR_COLOR = (255, 140, 40)
OFF_CENTER_COLOR = (60, 180, 255)
MIN_COLOR_RATIO = 0.42
MIN_RING_CONTRAST = 0.1
MIN_CIRCULARITY = 0.72
MAX_CIRCULARITY = 1.28
MIN_ASPECT_RATIO = 0.74
MAX_ASPECT_RATIO = 1.36
MIN_EXTENT = 0.56
MIN_SOLIDITY = 0.86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop tennis-ball detector for a MacBook webcam."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera number, image path, or video path. Use 0 for the default webcam.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the cleaned ball mask beside the camera view.",
    )
    parser.add_argument(
        "--min-area-scale",
        type=float,
        default=1.0,
        help="Raise this to ignore smaller detections, lower it to detect farther balls.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=DISPLAY_SCALE,
        help="Make the camera window larger or smaller. Try 1.0 or 1.25.",
    )
    parser.add_argument(
        "--center-tolerance",
        type=int,
        default=CENTER_TOLERANCE_PIXELS,
        help="How many pixels left or right of center still counts as centered.",
    )
    parser.add_argument(
        "--optimal-min-radius",
        type=int,
        default=OPTIMAL_MIN_RADIUS,
        help="Smallest ball radius that counts as good distance. Smaller means too far.",
    )
    parser.add_argument(
        "--optimal-max-radius",
        type=int,
        default=OPTIMAL_MAX_RADIUS,
        help="Largest ball radius that counts as good distance. Larger means too close.",
    )
    parser.add_argument(
        "--roundness",
        type=float,
        default=MIN_CIRCULARITY,
        help="Minimum roundness required. Higher rejects hands/background; lower finds imperfect balls.",
    )
    return parser.parse_args()


def source_value(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def open_source(source: str) -> cv2.VideoCapture:
    source_id = source_value(source)
    if isinstance(source_id, int):
        capture = open_camera(source_id, cv2.CAP_AVFOUNDATION)
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture

        capture.release()
        capture = open_camera(source_id, cv2.CAP_ANY)
    else:
        capture = cv2.VideoCapture(source_id)

    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return capture


def open_camera(source_id: int, backend: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source_id, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def make_ball_color_mask(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    color_masks = (
        hsv_range_mask(hue, saturation, value, 2, 28, 28, 24),  # orange
        hsv_range_mask(hue, saturation, value, 25, 88, 20, 38),  # yellow/light green
        hsv_range_mask(hue, saturation, value, 82, 108, 25, 35),  # light blue
        hsv_range_mask(hue, saturation, value, 100, 132, 35, 18),  # dark blue
        hsv_range_mask(hue, saturation, value, 112, 172, 10, 8),  # dark purple
        hsv_range_mask(hue, saturation, value, 0, 14, 45, 30),  # red/orange-red
        hsv_range_mask(hue, saturation, value, 166, 179, 35, 22),  # red/pink
    )
    mask = np.zeros(hue.shape, dtype=np.uint8)
    for color_mask in color_masks:
        mask = cv2.bitwise_or(mask, color_mask)

    skin_hue = cv2.inRange(hue, 0, 22)
    skin_saturation = cv2.inRange(saturation, 20, 115)
    skin_value = cv2.inRange(value, 80, 255)
    likely_skin = cv2.bitwise_and(cv2.bitwise_and(skin_hue, skin_saturation), skin_value)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(likely_skin))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def hsv_range_mask(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    hue_min: int,
    hue_max: int,
    saturation_min: int,
    value_min: int,
) -> np.ndarray:
    hue_mask = cv2.inRange(hue, hue_min, hue_max)
    saturation_mask = cv2.inRange(saturation, saturation_min, 255)
    value_mask = cv2.inRange(value, value_min, 255)
    return cv2.bitwise_and(cv2.bitwise_and(hue_mask, saturation_mask), value_mask)


def circle_color_ratio(mask: np.ndarray, center: tuple[int, int], radius: int) -> float:
    circle_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    circle_area = cv2.countNonZero(circle_mask)
    if circle_area == 0:
        return 0.0
    colored_pixels = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
    return colored_pixels / circle_area


def ring_color_ratio(mask: np.ndarray, center: tuple[int, int], radius: int) -> float:
    outer_radius = int(radius * 1.28)
    inner_radius = int(radius * 1.05)
    ring_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(ring_mask, center, outer_radius, 255, -1)
    cv2.circle(ring_mask, center, inner_radius, 0, -1)

    ring_area = cv2.countNonZero(ring_mask)
    if ring_area == 0:
        return 0.0

    colored_pixels = cv2.countNonZero(cv2.bitwise_and(mask, ring_mask))
    return colored_pixels / ring_area


def is_valid_ball_candidate(mask: np.ndarray, center: tuple[int, int], radius: int) -> tuple[bool, float, float]:
    inside_ratio = circle_color_ratio(mask, center, radius)
    outside_ratio = ring_color_ratio(mask, center, radius)
    has_enough_fill = inside_ratio >= MIN_COLOR_RATIO
    has_clean_boundary = inside_ratio - outside_ratio >= MIN_RING_CONTRAST and outside_ratio <= 0.35
    return has_enough_fill and has_clean_boundary, inside_ratio, outside_ratio


def contour_candidates(
    frame: np.ndarray,
    mask: np.ndarray,
    min_area_scale: float,
    optimal_min_radius: int,
    optimal_max_radius: int,
    min_circularity: float,
) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_center_x = frame.shape[1] / 2
    min_area = np.pi * (MIN_RADIUS_PIXELS**2) * min_area_scale
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        radius = int(max(w, h) / 2)
        if radius < MIN_RADIUS_PIXELS or radius > MAX_RADIUS_PIXELS:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
        aspect_ratio = w / h if h else 0
        extent = area / (w * h) if w and h else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area else 0

        if not (min_circularity <= circularity <= MAX_CIRCULARITY):
            continue
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue
        if extent < MIN_EXTENT:
            continue
        if solidity < MIN_SOLIDITY:
            continue

        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        is_valid, color_ratio, ring_ratio = is_valid_ball_candidate(mask, (center_x, center_y), radius)
        if not is_valid:
            continue

        detections.append(
            make_detection(
                center=(center_x, center_y),
                radius=radius,
                area=int(area),
                frame_center_x=frame_center_x,
                confidence=round(float(circularity * (color_ratio - ring_ratio)), 2),
                optimal_min_radius=optimal_min_radius,
                optimal_max_radius=optimal_max_radius,
            )
        )

    return detections


def make_detection(
    center: tuple[int, int],
    radius: int,
    area: int,
    frame_center_x: float,
    confidence: float,
    optimal_min_radius: int,
    optimal_max_radius: int,
) -> dict:
    x, y = center
    return {
        "name": "ball",
        "box": (x - radius, y - radius, radius * 2, radius * 2),
        "center": center,
        "area": area,
        "radius": radius,
        "x_error": int(x - frame_center_x),
        "confidence": confidence,
        "priority_score": int(radius * 1000 + area),
        "distance_zone": distance_zone(radius, optimal_min_radius, optimal_max_radius),
    }


def distance_zone(radius: int, optimal_min_radius: int, optimal_max_radius: int) -> str:
    if radius > optimal_max_radius:
        return "too_close"
    if radius < optimal_min_radius:
        return "too_far"
    return "optimal"


def border_color(detection: dict) -> tuple[int, int, int]:
    zone = detection["distance_zone"]
    if zone == "too_close":
        return TOO_CLOSE_COLOR
    if zone == "too_far":
        return TOO_FAR_COLOR
    return OPTIMAL_COLOR


def detections_overlap(first: dict, second: dict) -> bool:
    first_center = first["center"]
    second_center = second["center"]
    distance = np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    return distance < max(first["radius"], second["radius"]) * 0.75


def deduplicate_detections(detections: list[dict]) -> list[dict]:
    kept: list[dict] = []

    for detection in sorted(detections, key=lambda item: item["priority_score"], reverse=True):
        if any(detections_overlap(detection, kept_detection) for kept_detection in kept):
            continue
        kept.append(detection)

    kept.sort(key=lambda detection: detection["priority_score"], reverse=True)
    return kept[:MAX_DISPLAYED_DETECTIONS]


def assign_priorities(detections: list[dict]) -> None:
    for priority, detection in enumerate(detections, start=1):
        detection["priority"] = priority


class PriorityBallTracker:
    def __init__(self) -> None:
        self.detection: dict | None = None
        self.missed_frames = 0

    def update(
        self,
        detections: list[dict],
        optimal_min_radius: int,
        optimal_max_radius: int,
        frame_width: int,
    ) -> list[dict]:
        if not detections:
            return self._use_previous_if_recent()

        current = detections[0]
        if self.detection is None:
            self.detection = current.copy()
        else:
            self.detection = smooth_detection(
                self.detection,
                current,
                optimal_min_radius,
                optimal_max_radius,
                frame_width,
            )

        self.missed_frames = 0
        stabilized = [self.detection] + detections[1:]
        assign_priorities(stabilized)
        return stabilized

    def _use_previous_if_recent(self) -> list[dict]:
        if self.detection is None:
            return []

        self.missed_frames += 1
        if self.missed_frames > TRACKER_MAX_MISSED_FRAMES:
            self.detection = None
            return []

        held = self.detection.copy()
        held["held"] = True
        held["priority"] = 1
        return [held]


def smooth_detection(
    previous: dict,
    current: dict,
    optimal_min_radius: int,
    optimal_max_radius: int,
    frame_width: int,
) -> dict:
    alpha = TRACKER_SMOOTHING
    prev_x, prev_y = previous["center"]
    curr_x, curr_y = current["center"]
    center = (
        int(prev_x * (1 - alpha) + curr_x * alpha),
        int(prev_y * (1 - alpha) + curr_y * alpha),
    )
    radius = int(previous["radius"] * (1 - alpha) + current["radius"] * alpha)
    area = int(np.pi * radius * radius)

    smoothed = current.copy()
    smoothed.update(
        {
            "box": (center[0] - radius, center[1] - radius, radius * 2, radius * 2),
            "center": center,
            "area": area,
            "radius": radius,
            "x_error": int(center[0] - frame_width / 2),
            "priority_score": int(radius * 1000 + area),
            "distance_zone": distance_zone(radius, optimal_min_radius, optimal_max_radius),
            "held": False,
        }
    )
    return smoothed


def find_detections(
    frame: np.ndarray,
    min_area_scale: float,
    optimal_min_radius: int,
    optimal_max_radius: int,
    min_circularity: float,
) -> tuple[list[dict], np.ndarray]:
    mask = make_ball_color_mask(frame)
    detections = contour_candidates(
        frame,
        mask,
        min_area_scale,
        optimal_min_radius,
        optimal_max_radius,
        min_circularity,
    )

    detections = deduplicate_detections(detections)
    assign_priorities(detections)
    return detections, mask_from_detections(frame.shape[:2], detections)


def mask_from_detections(mask_shape: tuple[int, int], detections: list[dict]) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for detection in detections:
        cv2.circle(mask, detection["center"], detection["radius"], 255, -1)
    return mask


def steering_command(x_error: int, center_tolerance: int) -> str:
    if abs(x_error) <= center_tolerance:
        return "CENTERED"
    if x_error < 0:
        return "STEER LEFT"
    return "STEER RIGHT"


def distance_message(detection: dict) -> str:
    zone = detection["distance_zone"]
    if zone == "too_close":
        return "TOO CLOSE"
    if zone == "too_far":
        return "TOO FAR"
    return "GOOD DISTANCE"


def distance_action(detection: dict) -> str:
    zone = detection["distance_zone"]
    if zone == "too_close":
        return "MOVE FARTHER AWAY"
    if zone == "too_far":
        return "MOVE CLOSER"
    return "HOLD DISTANCE"


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_THICKNESS,
    )
    cv2.rectangle(
        frame,
        (x - 8, y - height - 10),
        (x + width + 8, y + baseline + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_big_arrow(frame: np.ndarray, command: str) -> None:
    _, width = frame.shape[:2]
    if command == "CENTERED":
        draw_text(frame, "CENTERED", (width // 2 - 125, 92), 1.0, OPTIMAL_COLOR)
        return

    if command == "STEER LEFT":
        label = "<- STEER LEFT"
        label_origin = (width // 2 - 175, 92)
    else:
        label = "STEER RIGHT ->"
        label_origin = (width // 2 - 185, 92)

    draw_text(frame, label, label_origin, 1.0, OFF_CENTER_COLOR)


def draw_centering_overlay(
    frame: np.ndarray,
    detections: list[dict],
    center_tolerance: int,
) -> None:
    height, width = frame.shape[:2]
    center_x = width // 2
    left_limit = center_x - center_tolerance
    right_limit = center_x + center_tolerance

    overlay = frame.copy()
    cv2.rectangle(overlay, (left_limit, 0), (right_limit, height), OPTIMAL_COLOR, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 2)
    cv2.line(frame, (left_limit, 0), (left_limit, height), OPTIMAL_COLOR, 2)
    cv2.line(frame, (right_limit, 0), (right_limit, height), OPTIMAL_COLOR, 2)

    if not detections:
        draw_text(frame, "No ball detected", (16, 45), HEADER_FONT_SCALE, (255, 255, 255))
        return

    target = detections[0]
    command = steering_command(target["x_error"], center_tolerance)
    draw_big_arrow(frame, command)

    status = (
        f"BALL #1 | {distance_message(target)} | {distance_action(target)}"
    )
    draw_text(frame, status, (16, 45), HEADER_FONT_SCALE, border_color(target))


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    center_tolerance: int,
) -> np.ndarray:
    annotated = frame.copy()
    draw_centering_overlay(annotated, detections, center_tolerance)

    for detection in detections:
        color = border_color(detection)
        center = detection["center"]
        radius = detection["radius"]
        thickness = BOX_THICKNESS if detection["priority"] == 1 else 2

        cv2.circle(annotated, center, radius, color, thickness)
        cv2.circle(annotated, center, 7, color, -1)

    draw_text(annotated, "q=quit", (16, annotated.shape[0] - 18), 0.62)
    return annotated


def resize_for_display(view: np.ndarray, display_scale: float) -> np.ndarray:
    if display_scale == 1.0:
        return view

    width = int(view.shape[1] * display_scale)
    height = int(view.shape[0] * display_scale)
    return cv2.resize(view, (width, height), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    args = parse_args()
    capture = open_source(args.source)
    tracker = PriorityBallTracker()

    print("Press q to quit.")
    print("Border color: red = too close, green = good distance, blue = too far.")
    print("Arrow feedback shows which way to steer to center the priority ball.")

    failed_reads = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            failed_reads += 1
            if failed_reads >= 10:
                print("Could not read frames from the camera.")
                print("Try: python object_detection_laptop.py --source 1 --debug")
                print("Also check macOS Camera permission for Terminal or your editor.")
                break
            continue
        failed_reads = 0

        detections, mask = find_detections(
            frame,
            args.min_area_scale,
            args.optimal_min_radius,
            args.optimal_max_radius,
            args.roundness,
        )
        detections = tracker.update(
            detections,
            args.optimal_min_radius,
            args.optimal_max_radius,
            frame.shape[1],
        )
        annotated = draw_detections(frame, detections, args.center_tolerance)

        if args.debug:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            view = np.hstack((annotated, mask_bgr))
        else:
            view = annotated

        view = resize_for_display(view, args.display_scale)
        cv2.imshow("Tennis Ball Detector", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
