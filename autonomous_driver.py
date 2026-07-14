from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


FRAME_CENTER_X = 320
CENTER_TOLERANCE_PIXELS = 50
MAX_STEERING = 0.55
SEARCH_STEERING = 0.32
FORWARD_SPEED = 0.22
TURNING_SPEED = 0.16
REVERSE_SPEED = -0.12
STOPPED = 0.0
VISION_TIMEOUT_SECONDS = 0.75


class TargetType(str, Enum):
    BALL = "ball"
    CONE = "cone"
    OBSTACLE = "obstacle"
    BUCKET = "bucket"
    UNKNOWN = "unknown"


class CourseMode(str, Enum):
    FIND_BALL = "find_ball"
    AVOID_OBSTACLE = "avoid_obstacle"
    FOLLOW_CONES = "follow_cones"
    AIM_AT_BUCKET = "aim_at_bucket"
    STOPPED = "stopped"


@dataclass
class Detection:
    name: TargetType
    x_error: int
    distance_zone: str = "unknown"
    radius: int = 0
    confidence: float = 1.0
    priority: int | None = None


@dataclass
class BallReport:
    priority_ball: Detection | None
    centered_balls: list[Detection]
    too_close_balls: list[Detection]
    too_far_balls: list[Detection]
    optimal_balls: list[Detection]


@dataclass
class DriveCommand:
    steering: float
    throttle: float
    mode: CourseMode
    reason: str
    ball_report: BallReport | None = None


class TraxxasPrintController:
    """Temporary controller that prints what the Traxxas car would be told."""

    def send(self, command: DriveCommand) -> None:
        report = ball_report_message(command.ball_report)
        print(
            f"{command.mode.value:15} "
            f"steering={command.steering:+.2f} "
            f"throttle={command.throttle:+.2f} "
            f"reason={command.reason}"
            f"{report}"
        )


class AutonomousDriver:
    def __init__(self) -> None:
        self.last_seen_time: float | None = None
        self.previous_command = DriveCommand(
            steering=0.0,
            throttle=0.0,
            mode=CourseMode.STOPPED,
            reason="not started",
        )

    def choose_command(
        self,
        detections: Iterable[Detection],
        course_mode: CourseMode = CourseMode.FIND_BALL,
        now: float | None = None,
    ) -> DriveCommand:
        now = time.monotonic() if now is None else now
        detections = list(detections)

        if detections:
            self.last_seen_time = now
        elif self._vision_is_stale(now):
            return self._remember(DriveCommand(0.0, 0.0, course_mode, "vision stale, stopping"))

        if course_mode == CourseMode.STOPPED:
            return self._remember(DriveCommand(0.0, 0.0, course_mode, "manual stop mode"))

        ball_report = self._analyze_balls(detections)
        obstacle = self._first_of_type(detections, TargetType.OBSTACLE)
        if obstacle is not None and obstacle.distance_zone == "too_close":
            return self._remember(self._avoid_obstacle(obstacle, course_mode, ball_report))

        if course_mode == CourseMode.FOLLOW_CONES:
            return self._remember(self._follow_cone(detections))

        if course_mode == CourseMode.AIM_AT_BUCKET:
            return self._remember(self._aim_at_bucket(detections))

        if ball_report.priority_ball is not None:
            return self._remember(self._drive_to_priority_ball(ball_report, course_mode))

        return self._remember(
            DriveCommand(
                steering=SEARCH_STEERING,
                throttle=0.0,
                mode=course_mode,
                reason="target not visible, searching in place",
            )
        )

    def _drive_to_priority_ball(self, ball_report: BallReport, course_mode: CourseMode) -> DriveCommand:
        ball = ball_report.priority_ball
        if ball is None:
            return DriveCommand(SEARCH_STEERING, 0.0, course_mode, "no priority ball", ball_report)

        steering = steering_from_error(ball.x_error)
        centered = abs(ball.x_error) <= CENTER_TOLERANCE_PIXELS
        label = priority_label(ball)

        if ball.distance_zone == "too_close":
            return DriveCommand(
                0.0,
                REVERSE_SPEED,
                course_mode,
                f"{label} too close, backing up slowly",
                ball_report,
            )

        if not centered:
            return DriveCommand(
                steering,
                TURNING_SPEED,
                course_mode,
                f"centering {label}",
                ball_report,
            )

        if ball.distance_zone == "optimal":
            return DriveCommand(
                0.0,
                STOPPED,
                course_mode,
                f"{label} centered and at good distance",
                ball_report,
            )

        return DriveCommand(
            0.0,
            FORWARD_SPEED,
            course_mode,
            f"{label} centered but too far, driving forward",
            ball_report,
        )

    def _avoid_obstacle(
        self,
        obstacle: Detection,
        course_mode: CourseMode,
        ball_report: BallReport | None = None,
    ) -> DriveCommand:
        if obstacle.x_error < 0:
            steering = MAX_STEERING
            reason = "obstacle on left, steering right"
        else:
            steering = -MAX_STEERING
            reason = "obstacle on right, steering left"

        return DriveCommand(steering, TURNING_SPEED, course_mode, reason, ball_report)

    def _follow_cone(self, detections: list[Detection]) -> DriveCommand:
        cone = self._first_of_type(detections, TargetType.CONE)
        if cone is None:
            return DriveCommand(SEARCH_STEERING, 0.0, CourseMode.FOLLOW_CONES, "no cone visible, searching")

        return DriveCommand(
            steering_from_error(cone.x_error),
            TURNING_SPEED,
            CourseMode.FOLLOW_CONES,
            "following cone line",
        )

    def _aim_at_bucket(self, detections: list[Detection]) -> DriveCommand:
        bucket = self._first_of_type(detections, TargetType.BUCKET)
        if bucket is None:
            return DriveCommand(SEARCH_STEERING, 0.0, CourseMode.AIM_AT_BUCKET, "no bucket visible, searching")

        if abs(bucket.x_error) <= 35:
            return DriveCommand(0.0, 0.0, CourseMode.AIM_AT_BUCKET, "bucket centered, ready to shoot")

        return DriveCommand(
            steering_from_error(bucket.x_error),
            0.0,
            CourseMode.AIM_AT_BUCKET,
            "turning to center bucket",
        )

    def _vision_is_stale(self, now: float) -> bool:
        if self.last_seen_time is None:
            return True
        return now - self.last_seen_time > VISION_TIMEOUT_SECONDS

    def _analyze_balls(self, detections: list[Detection]) -> BallReport:
        balls = [detection for detection in detections if detection.name == TargetType.BALL]
        priority_ball = choose_priority_ball(balls)
        return BallReport(
            priority_ball=priority_ball,
            centered_balls=[
                ball for ball in balls if abs(ball.x_error) <= CENTER_TOLERANCE_PIXELS
            ],
            too_close_balls=[
                ball for ball in balls if ball.distance_zone == "too_close"
            ],
            too_far_balls=[
                ball for ball in balls if ball.distance_zone == "too_far"
            ],
            optimal_balls=[
                ball for ball in balls if ball.distance_zone == "optimal"
            ],
        )

    def _remember(self, command: DriveCommand) -> DriveCommand:
        command = smooth_command(self.previous_command, command)
        self.previous_command = command
        return command

    @staticmethod
    def _first_of_type(detections: list[Detection], target_type: TargetType) -> Detection | None:
        return next((detection for detection in detections if detection.name == target_type), None)


def steering_from_error(x_error: int) -> float:
    steering = x_error / FRAME_CENTER_X
    return clamp(steering, -MAX_STEERING, MAX_STEERING)


def choose_priority_ball(balls: list[Detection]) -> Detection | None:
    if not balls:
        return None

    return min(
        balls,
        key=lambda ball: (
            ball.priority if ball.priority is not None else 999,
            -ball.radius,
            abs(ball.x_error),
        ),
    )


def priority_label(ball: Detection) -> str:
    if ball.priority is None:
        return "priority ball"
    return f"priority ball #{ball.priority}"


def ball_report_message(report: BallReport | None) -> str:
    if report is None or report.priority_ball is None:
        return ""

    ball = report.priority_ball
    return (
        f" | {priority_label(ball)}: "
        f"x_error={ball.x_error:+d}, "
        f"distance={ball.distance_zone}, "
        f"centered={abs(ball.x_error) <= CENTER_TOLERANCE_PIXELS}; "
        f"balls close={len(report.too_close_balls)}, "
        f"far={len(report.too_far_balls)}, "
        f"centered={len(report.centered_balls)}"
    )


def smooth_command(previous: DriveCommand, current: DriveCommand) -> DriveCommand:
    steering = previous.steering * 0.35 + current.steering * 0.65
    throttle = previous.throttle * 0.35 + current.throttle * 0.65
    return DriveCommand(
        steering=clamp(steering, -MAX_STEERING, MAX_STEERING),
        throttle=clamp(throttle, -0.25, 0.25),
        mode=current.mode,
        reason=current.reason,
        ball_report=current.ball_report,
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def detection_from_dict(data: dict) -> Detection:
    name = data.get("name", TargetType.UNKNOWN)
    try:
        target_type = TargetType(name)
    except ValueError:
        target_type = TargetType.UNKNOWN

    raw_priority = data.get("priority")
    priority = int(raw_priority) if raw_priority is not None else None

    return Detection(
        name=target_type,
        x_error=int(data.get("x_error", 0)),
        distance_zone=data.get("distance_zone", "unknown"),
        radius=int(data.get("radius", 0)),
        confidence=float(data.get("confidence", 1.0)),
        priority=priority,
    )


def demo_scenarios() -> list[tuple[CourseMode, list[Detection]]]:
    return [
        (CourseMode.FIND_BALL, []),
        (CourseMode.FIND_BALL, [Detection(TargetType.BALL, -150, "too_far", 20, priority=1)]),
        (CourseMode.FIND_BALL, [Detection(TargetType.BALL, 90, "too_far", 24, priority=1)]),
        (
            CourseMode.FIND_BALL,
            [
                Detection(TargetType.BALL, 140, "too_far", 18, priority=2),
                Detection(TargetType.BALL, -35, "too_far", 36, priority=1),
                Detection(TargetType.BALL, 10, "too_close", 70, priority=3),
            ],
        ),
        (CourseMode.FIND_BALL, [Detection(TargetType.BALL, 0, "optimal", 45, priority=1)]),
        (
            CourseMode.FIND_BALL,
            [
                Detection(TargetType.BALL, 20, "too_far", 30, priority=1),
                Detection(TargetType.OBSTACLE, -60, "too_close", 70),
            ],
        ),
        (CourseMode.FOLLOW_CONES, [Detection(TargetType.CONE, 110, "unknown", 36)]),
        (CourseMode.AIM_AT_BUCKET, [Detection(TargetType.BUCKET, -80, "unknown", 52)]),
        (CourseMode.AIM_AT_BUCKET, [Detection(TargetType.BUCKET, 12, "unknown", 60)]),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Traxxas autonomy demo. It prints steering and throttle instead of moving motors."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run sample obstacle-course detections through the driver.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = TraxxasPrintController()
    driver = AutonomousDriver()

    if not args.demo:
        print("Run with: python autonomous_driver.py --demo")
        return

    for mode, detections in demo_scenarios():
        command = driver.choose_command(detections, mode)
        controller.send(command)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
