# RC Car Tennis Ball Detection Starter

This project is a beginner-friendly starting point for the vision part of an autonomous RC car. The current code runs on a MacBook webcam and detects tennis balls before we move anything to the Raspberry Pi.

The final robot can be built in stages:

1. Drive by remote control.
2. Detect tennis balls.
3. Use detection results to help the driver.
4. Use detection results to steer automatically.
5. Add ball intake and shooting.

The code in this folder runs on a laptop first, using a webcam, image, or video file. That lets you test the vision logic before moving it to the Raspberry Pi 5 and Arducam IMX708.

## What We Are Doing First

We are not using a trained AI model yet. We are starting with simple camera tracking because it is easier to understand and easier to debug.

Each webcam frame is just a picture. The program:

1. Reads one image from the webcam.
2. Builds a broad colored-object mask so gray classroom objects are mostly ignored.
3. Looks for round, filled-in shapes that are the right size for tennis balls.
4. Draws a simple circle, center point, distance status, and steering cue on the screen.
5. Gives closer balls higher priority by using the ball's size in the image.

`x_error` is especially important for the future RC car:

- Negative `x_error`: the ball is left of the camera center.
- Positive `x_error`: the ball is right of the camera center.
- Near zero: the ball is centered.

The number is measured in camera pixels. At the faster 640x480 camera setting, the center of the image is around x=320. If the ball center is at x=390, then `x_error` is about `+70`. If the ball center is at x=250, then `x_error` is about `-70`.

Later, steering can use this value.

The program also ranks balls by priority. A ball that looks larger in the camera is probably closer to the robot, so it becomes the higher-priority target. The top of the camera view shows `PRIORITY #1`, which is the ball the robot would focus on first. Small noisy color patches are filtered out before priority is assigned.

## System Sections

### 1. Mechanical

- RC chassis, motors, steering, battery, and mounting plates.
- Camera mount near the front of the car.
- Add a small front intake and shooter after basic navigation works.

Keep the ball system compact. The goal is not a large turret or box on top of
the RC car. Treat it like a short one-ball path: touch the ball, pull it in,
guide it a few inches, and shoot it. Avoid extra storage, tall towers, and wide
scoops unless testing proves they are needed.

Condensed intake direction:

- Use two short side plates at the nose of the car.
- Put one soft roller or compliant wheel shaft between the plates.
- Make the intake just wider than one tennis ball.
- Angle the roller down/back so it pulls the ball onto a small ramp.
- Let the ramp feed directly into the shooter instead of using a separate
  holding box.

Shooter ideas for a tight car when the intake uses the front space:

- Single top wheel shooter: one fast wheel above the ball, with a smooth ramp
  underneath. This is the simplest first version because the intake ramp can be
  the shooter ramp too.
- Two vertical side wheels: one wheel on the left and one on the right pinch
  the ball and shoot it forward. This can be short front-to-back, but it needs
  room across the width of the car.
- Offset top wheel: place the shooter wheel slightly behind the intake roller,
  not above the very front. The ball enters from the front but launches from
  closer to the middle of the chassis.
- Angled top wheel: tilt the wheel and exit ramp upward so the ball lobs toward
  the bucket without needing a tall arm.
- Rear-fed mini shooter: intake at the front, guide the ball under/behind the
  camera mount, then shoot from the center where there is usually more room.
- Side-exit shooter: feed the ball in from the front, turn it slightly with a
  curved guide, and shoot out at a small angle. This can help if the front is
  blocked by the camera or steering parts.
- Under-camera shooter: mount the camera higher on a light bracket and keep the
  intake/shooter path low underneath it.
- Flywheel plus backstop: use one powered wheel and one fixed curved wall
  instead of two powered wheels. This saves motor space and wiring.
- Servo release into wheel: hold one ball in a tiny pocket, spin the wheel up,
  then use a small servo gate to let the ball touch the wheel only when aimed.
- Low lob ramp: use a slower wheel and a curved ramp to toss the ball upward
  instead of firing it hard. This may be easier to package and control.

For a first build, the best layout is probably:

```text
front of car
-> small roller intake
-> short curved ramp / shooter floor
-> single top wheel mounted slightly behind the intake
-> ball exits upward/forward
```

That keeps the mechanism narrow, low, and removable. It also leaves room for
the camera, Raspberry Pi, battery, and wiring. The first prototype should only
handle one ball at a time; add storage later only if the course requires it.

### 2. Electronics

- Raspberry Pi 5 for vision and high-level decisions.
- Arducam IMX708 camera for live video.
- Motor controller or servo controller for throttle and steering.
- Separate safe power paths for motors and Raspberry Pi when possible.

### 3. Vision

- Start with color detection for cones, balls, and bucket targets.
- Move to trained object detection after the course objects are finalized.
- The most useful output is not just "object found"; it is also where the object is in the frame.

### 4. Driving Logic

The camera image has a center line. For each detected object, the code reports `x_error`:

- Negative `x_error`: object is left of center.
- Positive `x_error`: object is right of center.
- Near zero: object is centered.

Later, the car can use this value to steer.

The file `autonomous_driver.py` is a safe starting point for the Traxxas driving
logic. It does not move motors yet. It reads example detections for balls,
cones, obstacles, and the bucket, then prints the steering and throttle values
that would eventually be sent to the Traxxas steering servo and ESC.

The driver also tracks multiple balls. It picks the priority ball, reports
whether that ball is centered, counts balls that are too close or too far, and
centers the priority ball before driving forward.

Run the demo with:

```bash
python autonomous_driver.py --demo
```

This lets the driving logic be tested separately while the hardware and vision
parts are still being built.

### 5. Autonomy Milestones

- Follow one cone or colored ball slowly.
- Turn away from one obstacle.
- Slalom between cones.
- Detect bucket.
- Drive to ball.
- Intake ball.
- Aim at bucket.
- Shoot ball.
- Combine all steps into a full course routine.

## MacBook Setup

Install Python 3.11 or newer if you do not already have Python installed.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

Run with your laptop webcam:

```bash
python object_detection_laptop.py --source 0
```

Run with debug view:

```bash
python object_detection_laptop.py --source 0 --debug
```

If the webcam does not open, try camera 1:

```bash
python object_detection_laptop.py --source 1
```

Press `q` to quit.

On macOS, you may be asked to allow Terminal or your code editor to use the camera. Choose allow.

## First Test

Start with one tennis ball on a plain background.

1. Hold up one ball.
2. Check that the program draws a circle around it.
3. Move it left and right.
4. Watch `x_error` become negative on the left and positive on the right.
5. Move it closer and farther away.
6. Watch the border change color.
7. Try a second ball.

Then test multiple balls at the same time.

## Centering A Ball

The camera view has a center guide overlay:

- The white vertical line is the exact center of the image.
- The green shaded area is the "centered enough" zone.
- `CENTERED` means the highest-priority detected ball is inside that zone.
- `STEER LEFT` or `STEER RIGHT` tells the driver or robot which way to turn so the priority ball moves toward the center.
- One steering cue shows the direction to turn.

The default centered range is 80 pixels to the left or right of center. To make the range stricter:

```bash
python object_detection_laptop.py --source 0 --debug --center-tolerance 40
```

To make the range more forgiving:

```bash
python object_detection_laptop.py --source 0 --debug --center-tolerance 120
```

## What The Starter Detects

The starter detects balls without trying to identify their color. Each candidate must be:

- colored enough to stand out from gray classroom objects
- inside one of the broad tennis-ball color bands, including orange, light green, light blue, dark blue, dark purple, red, and pink
- large enough to be a real ball
- round enough to be ball-shaped
- filled in enough to avoid thin lines, shadows, and chair edges
- separated enough from the surrounding area to avoid clothing, skin, and background patches
- smooth enough around the outline to reject fingers and hand shapes
- solid enough inside the circle to reject hands that only have a curved edge

The detector only displays the strongest few detections so the screen does not get flooded with noise.

Orange and dark purple have wider color bands because webcams often see them as dimmer or less saturated than light blue.

The border color shows distance:

- Red: too close, move farther away
- Green: good distance, hold distance
- Blue: too far, move closer

The priority ball is smoothed across frames so the circle does not flash on and off when one camera frame briefly misses the ball.

The green range is intentionally narrow. By default, the ball is green only when its camera radius is between 34 and 58 pixels.

To make the good-distance range even smaller:

```bash
python object_detection_laptop.py --source 0 --debug --optimal-min-radius 40 --optimal-max-radius 52
```

To make the good-distance range wider again:

```bash
python object_detection_laptop.py --source 0 --debug --optimal-min-radius 28 --optimal-max-radius 70
```

## If Detection Is Messy

Use debug view:

```bash
python object_detection_laptop.py --source 0 --debug
```

The right side of the window shows the cleaned ball mask. White means "the program thinks this area is a ball." Black means "ignore this area."

Try these fixes:

- Use a plain background.
- Avoid direct sunlight or harsh reflections.
- Keep the ball 1 to 6 feet from the webcam.
- Raise the area threshold if random spots are detected:

```bash
python object_detection_laptop.py --source 0 --debug --min-area-scale 1.5
```

For very noisy rooms, make it stricter:

```bash
python object_detection_laptop.py --source 0 --debug --min-area-scale 2.0
```

If it detects hands or other non-round objects, make the roundness requirement stricter:

```bash
python object_detection_laptop.py --source 0 --debug --roundness 0.85
```

If it starts missing real balls, loosen it slightly:

```bash
python object_detection_laptop.py --source 0 --debug --roundness 0.68
```

If the light green ball is missed, use a plain background first, then try:

```bash
python object_detection_laptop.py --source 0 --debug --roundness 0.68
```

- Lower the area threshold if the ball is too far away to detect:

```bash
python object_detection_laptop.py --source 0 --debug --min-area-scale 0.6
```

## Testing With Saved Files

Run on a saved video:

```bash
python object_detection_laptop.py --source path/to/video.mp4
```

Run on a saved image:

```bash
python object_detection_laptop.py --source path/to/image.jpg
```

Press `q` to quit.

## Important Beginner Notes

Color detection is simpler than AI object detection, but it is sensitive to lighting. Test in lighting similar to the real obstacle course.

For a real competition-style course, a custom trained model is usually better. That means collecting images of your own cones, balls, bucket, and background, labeling them, training a model such as YOLO, then running that model on the Raspberry Pi.

## Moving To Raspberry Pi Later

The laptop code uses OpenCV, which can also run on Raspberry Pi. The main difference is the camera input. On the Pi, you use Picamera2 to read from the Arducam IMX708, then pass each frame into the same detection functions.

The flow on the robot is:

```text
Arducam IMX708 frame
-> ball detection
-> priority ball
-> steering/throttle decision
-> motor or servo controller
```

The most important detection values are:

- `x_error`: how far left or right the priority ball is from the center of the camera
- `distance_zone`: `too_far`, `optimal`, or `too_close`
- `radius`: how large the ball appears, which is a rough distance estimate

Example robot decisions:

```text
No ball detected: rotate/search slowly
Ball left of center: steer left
Ball right of center: steer right
Ball centered and too far: drive forward
Ball centered and optimal: stop or start intake
Ball centered and too close: back up
```

On Raspberry Pi, keep vision and motor control separate at first. Print the planned command before sending power to motors:

```python
if not detections:
    command = "SEARCH"
else:
    target = detections[0]
    if target["x_error"] < -50:
        command = "STEER_LEFT"
    elif target["x_error"] > 50:
        command = "STEER_RIGHT"
    elif target["distance_zone"] == "too_far":
        command = "DRIVE_FORWARD"
    elif target["distance_zone"] == "too_close":
        command = "BACK_UP"
    else:
        command = "STOP_OR_INTAKE"
```

After the printed commands look correct, connect them to your motor hardware. For many RC-style builds, the Raspberry Pi does not drive motors directly. It sends low-power control signals to a motor controller, ESC, servo controller, or microcontroller. Always test with the wheels lifted first.

Keep the project staged:

1. Make detection work on laptop.
2. Make the camera work on Raspberry Pi.
3. Run the same detection logic on Raspberry Pi.
4. Print steering decisions without moving motors.
5. Move motors slowly with the wheels lifted.
6. Test on the floor at low speed.
