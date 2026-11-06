
import math
import random
import threading
import time

import cv2
import mediapipe as mp
import pygame


# =========================================
# SETTINGS
# =========================================

WIDTH, HEIGHT = 900, 740
FPS = 60

ROAD_LEFT = 175
ROAD_RIGHT = 725

PLAYER_Y = HEIGHT - 145

CAR_W = 62
CAR_H = 108

PREVIEW_W = 272
PREVIEW_H = 153

WHITE = (243, 248, 255)
CYAN = (58, 220, 235)
YELLOW = (255, 209, 89)
RED = (239, 83, 81)
DARK = (13, 21, 33)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# =========================================
# VIRTUAL STEERING WHEEL
# =========================================

def draw_steering_wheel(frame, left, right, angle):

    x1, y1 = left
    x2, y2 = right

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    distance = math.hypot(x2 - x1, y2 - y1)

    radius = int(
        clamp(distance * 0.50, 40, 145)
    )

    aa = cv2.LINE_AA

    # Transparent background
    overlay = frame.copy()

    cv2.circle(
        overlay,
        (cx, cy),
        radius - 5,
        (18, 24, 31),
        -1,
        aa
    )

    cv2.addWeighted(
        overlay,
        0.35,
        frame,
        0.65,
        0,
        frame
    )

    # Steering wheel shadow
    cv2.circle(
        frame,
        (cx + 2, cy + 3),
        radius,
        (5, 9, 15),
        17,
        aa
    )

    # Rubber ring
    cv2.circle(
        frame,
        (cx, cy),
        radius,
        (32, 37, 43),
        14,
        aa
    )

    # Metallic highlight
    cv2.circle(
        frame,
        (cx, cy),
        radius,
        (230, 224, 208),
        3,
        aa
    )

    # Inner ring
    cv2.circle(
        frame,
        (cx, cy),
        int(radius * 0.78),
        (68, 85, 99),
        2,
        aa
    )

    base = math.radians(angle)

    # Three rotating spokes
    for offset in (
        0,
        math.pi,
        math.pi / 2
    ):

        radians = base + offset

        end = (
            int(
                cx
                + math.cos(radians)
                * radius * 0.80
            ),
            int(
                cy
                + math.sin(radians)
                * radius * 0.80
            )
        )

        cv2.line(
            frame,
            (cx, cy),
            end,
            (8, 14, 21),
            16,
            aa
        )

        cv2.line(
            frame,
            (cx, cy),
            end,
            (85, 99, 112),
            9,
            aa
        )

        cv2.line(
            frame,
            (cx, cy),
            end,
            (204, 205, 207),
            2,
            aa
        )

    # Left and right grip indicators
    for offset in (0, math.pi):

        radians = base + offset

        grip = (
            int(
                cx
                + math.cos(radians) * radius
            ),
            int(
                cy
                + math.sin(radians) * radius
            )
        )

        cv2.circle(
            frame,
            grip,
            10,
            (30, 235, 250),
            -1,
            aa
        )

        cv2.circle(
            frame,
            grip,
            5,
            (252, 255, 255),
            -1,
            aa
        )

    # Steering wheel center
    cv2.circle(
        frame,
        (cx, cy),
        int(radius * 0.22),
        (12, 17, 23),
        -1,
        aa
    )

    cv2.circle(
        frame,
        (cx, cy),
        int(radius * 0.19),
        (69, 216, 234),
        3,
        aa
    )

    cv2.putText(
        frame,
        "SR",
        (cx - 13, cy + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (245, 250, 255),
        2,
        aa
    )


# =========================================
# CAMERA CONTROLLER
# =========================================

class CameraWheel:

    def __init__(self):

        self.lock = threading.Lock()

        self.running = False
        self.thread = None

        self.active = False

        self.steer = 0.0
        self.angle = 0.0
        self.center_angle = 0.0

        self.hands = 0
        self.last_seen = 0.0

        self.status = "Starting camera..."
        self.preview = None

        self.visual_center = None
        self.visual_angle = None

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    def calibrate(self):

        with self.lock:

            recent = (
                time.monotonic()
                - self.last_seen < 0.6
            )

            if self.hands == 2 and recent:

                self.center_angle = self.angle
                self.steer = 0.0

                return True

        return False

    def snapshot(self):

        with self.lock:

            valid = (
                self.active
                and time.monotonic()
                - self.last_seen < 0.6
            )

            return (
                valid,
                self.steer,
                self.hands,
                self.status,
                self.preview
            )

    def close(self):

        self.running = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(timeout=1.5)

    # =====================================
    # CAMERA WORKER
    # =====================================

    def _worker(self):

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():

            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            with self.lock:

                self.active = False
                self.status = "Camera not found"

            return

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            360
        )

        try:

            with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.50
            ) as detector:

                while self.running:

                    success, frame = cap.read()

                    if not success:

                        with self.lock:

                            self.active = False
                            self.hands = 0
                            self.status = (
                                "Camera unavailable"
                            )

                        time.sleep(0.10)
                        continue

                    # Mirror webcam
                    frame = cv2.flip(
                        cv2.resize(
                            frame,
                            (640, 360)
                        ),
                        1
                    )

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    rgb.flags.writeable = False

                    result = detector.process(rgb)

                    # Hand positions
                    points = []

                    if result.multi_hand_landmarks:

                        for hand in (
                            result.multi_hand_landmarks[:2]
                        ):

                            landmarks = hand.landmark

                            px = (
                                0.55 * landmarks[0].x
                                + 0.45 * landmarks[9].x
                            )

                            py = (
                                0.55 * landmarks[0].y
                                + 0.45 * landmarks[9].y
                            )

                            points.append((
                                int(px * 640),
                                int(py * 360)
                            ))

                    detected = len(points)

                    # =================================
                    # BOTH HANDS DETECTED
                    # =================================

                    if detected == 2:

                        points.sort(
                            key=lambda p: p[0]
                        )

                        x1, y1 = points[0]
                        x2, y2 = points[1]

                        spacing = math.hypot(
                            x2 - x1,
                            y2 - y1
                        )

                        if spacing >= 45:

                            angle = math.degrees(
                                math.atan2(
                                    y2 - y1,
                                    x2 - x1
                                )
                            )

                            middle = (
                                (x1 + x2) / 2,
                                (y1 + y2) / 2
                            )

                            # Smooth steering wheel
                            if self.visual_center is None:

                                self.visual_center = middle
                                self.visual_angle = angle

                            else:

                                self.visual_center = (
                                    self.visual_center[0] * 0.65
                                    + middle[0] * 0.35,

                                    self.visual_center[1] * 0.65
                                    + middle[1] * 0.35
                                )

                                self.visual_angle = (
                                    self.visual_angle * 0.65
                                    + angle * 0.35
                                )

                            center = self.visual_center

                            half = spacing / 2

                            radians = math.radians(
                                self.visual_angle
                            )

                            smoothed_left = (
                                int(
                                    center[0]
                                    - half * math.cos(radians)
                                ),
                                int(
                                    center[1]
                                    - half * math.sin(radians)
                                )
                            )

                            smoothed_right = (
                                int(
                                    center[0]
                                    + half * math.cos(radians)
                                ),
                                int(
                                    center[1]
                                    + half * math.sin(radians)
                                )
                            )

                            # Draw steering wheel
                            draw_steering_wheel(
                                frame,
                                smoothed_left,
                                smoothed_right,
                                self.visual_angle
                            )

                            # Calculate steering
                            with self.lock:

                                self.angle = angle

                                raw = clamp(
                                    (
                                        angle
                                        - self.center_angle
                                    ) / 32.0,
                                    -1.0,
                                    1.0
                                )

                                if abs(raw) < 0.08:
                                    raw = 0.0

                                self.steer = (
                                    0.68 * self.steer
                                    + 0.32 * raw
                                )

                                self.active = True

                                self.last_seen = (
                                    time.monotonic()
                                )

                                self.status = (
                                    "Virtual steering wheel ACTIVE"
                                )

                        else:

                            with self.lock:

                                self.active = False
                                self.steer = 0.0
                                self.status = (
                                    "Move hands farther apart"
                                )

                            self.visual_center = None
                            self.visual_angle = None

                    else:

                        with self.lock:

                            self.active = False
                            self.steer = 0.0
                            self.status = (
                                "Show TWO hands"
                            )

                        self.visual_center = None
                        self.visual_angle = None

                    # Draw small tracking markers
                    for point in points:

                        cv2.circle(
                            frame,
                            point,
                            7,
                            (22, 232, 250),
                            -1,
                            cv2.LINE_AA
                        )

                        cv2.circle(
                            frame,
                            point,
                            11,
                            (248, 251, 253),
                            2,
                            cv2.LINE_AA
                        )

                    # Camera status
                    cv2.putText(
                        frame,
                        f"HANDS {detected}/2",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    # Camera preview
                    small = cv2.resize(
                        frame,
                        (PREVIEW_W, PREVIEW_H)
                    )

                    preview = cv2.cvtColor(
                        small,
                        cv2.COLOR_BGR2RGB
                    ).tobytes()

                    with self.lock:

                        self.hands = detected
                        self.preview = preview

        except Exception as error:

            print("Camera error:", error)

            with self.lock:

                self.active = False
                self.status = "Camera error"

        finally:

            cap.release()


# =========================================
# TEXT
# =========================================

def draw_text(
    screen,
    font,
    text,
    x,
    y,
    paint=WHITE,
    center=False
):

    image = font.render(
        text,
        True,
        paint
    )

    if center:

        rect = image.get_rect(
            center=(x, y)
        )

    else:

        rect = (x, y)

    screen.blit(image, rect)


# =========================================
# DRAW CAR
# =========================================

def draw_car(
    screen,
    x,
    y,
    paint,
    player=False
):

    x = int(x)
    y = int(y)

    rect = pygame.Rect(
        x - CAR_W // 2,
        y - CAR_H // 2,
        CAR_W,
        CAR_H
    )

    # Shadow
    pygame.draw.rect(
        screen,
        (12, 19, 28),
        rect.move(5, 7),
        border_radius=12
    )

    # Wheels
    for dx in (
        -CAR_W // 2 - 7,
        CAR_W // 2 - 1
    ):

        for dy in (-31, 26):

            pygame.draw.rect(
                screen,
                (10, 14, 19),
                (x + dx, y + dy, 8, 29),
                border_radius=3
            )

    # Car body
    pygame.draw.rect(
        screen,
        paint,
        rect,
        border_radius=12
    )

    # Cabin
    pygame.draw.rect(
        screen,
        (34, 54, 70),
        (x - 24, y - 18, 48, 38),
        border_radius=8
    )

    # Front windshield
    pygame.draw.rect(
        screen,
        (105, 189, 216),
        (x - 22, y - 38, 44, 22),
        border_radius=6
    )

    # Rear windshield
    pygame.draw.rect(
        screen,
        (31, 52, 72),
        (x - 22, y + 23, 44, 13),
        border_radius=3
    )

    # Lights
    for dx in (-22, 16):

        pygame.draw.rect(
            screen,
            YELLOW,
            (x + dx, y - 49, 7, 9),
            border_radius=2
        )

        pygame.draw.rect(
            screen,
            RED,
            (x + dx, y + 43, 7, 7),
            border_radius=2
        )

    # Player spoiler
    if player:

        pygame.draw.rect(
            screen,
            (15, 38, 50),
            (x - 35, y + 35, 70, 7),
            border_radius=2
        )

    return rect.inflate(-12, -16)


# =========================================
# ROAD
# =========================================

def draw_road(screen, scroll):

    # Grass
    screen.fill((30, 112, 78))

    # Asphalt
    pygame.draw.rect(
        screen,
        (26, 35, 47),
        (
            ROAD_LEFT,
            0,
            ROAD_RIGHT - ROAD_LEFT,
            HEIGHT
        )
    )

    # Road borders
    for x in (
        ROAD_LEFT - 12,
        ROAD_RIGHT
    ):

        pygame.draw.rect(
            screen,
            (153, 157, 163),
            (x, 0, 12, HEIGHT)
        )

        for y in range(
            -50,
            HEIGHT + 60,
            70
        ):

            py = int(
                (
                    y + scroll * 1.3
                ) % (HEIGHT + 70) - 35
            )

            pygame.draw.rect(
                screen,
                (240, 236, 220),
                (x, py, 12, 36)
            )

    # Lane markings
    for lane in (1, 2):

        x = (
            ROAD_LEFT
            + (ROAD_RIGHT - ROAD_LEFT)
            * lane // 3
        )

        for y in range(
            -90,
            HEIGHT + 90,
            98
        ):

            py = int(
                (
                    y + scroll
                ) % (HEIGHT + 98) - 50
            )

            pygame.draw.rect(
                screen,
                (215, 222, 220),
                (x - 3, py, 6, 55),
                border_radius=2
            )

    # Trees
    for x in (80, 820):

        for k in range(8):

            y = int(
                (
                    k * 125
                    + scroll * 0.65
                ) % (HEIGHT + 125) - 75
            )

            pygame.draw.circle(
                screen,
                (17, 80, 56),
                (x + 5, y + 5),
                21
            )

            pygame.draw.circle(
                screen,
                (54, 164, 104),
                (x, y),
                19
            )


# =========================================
# MAIN GAME
# =========================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "STEERING RACE 2D | CAMERA WHEEL"
    )

    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont(
        "arial",
        53,
        bold=True
    )

    font_mid = pygame.font.SysFont(
        "arial",
        26,
        bold=True
    )

    font_small = pygame.font.SysFont(
        "arial",
        19
    )

    font_tiny = pygame.font.SysFont(
        "arial",
        16
    )

    wheel = CameraWheel()
    wheel.start()

    camera_mode = True
    reverse = False

    state = "menu"

    player_x = WIDTH / 2

    lives = 3
    score = 0
    distance = 0.0

    speed = 330.0
    scroll = 0.0
    spawn_timer = 1.2

    obstacles = []

    notice = ""
    notice_until = 0.0

    # =====================================
    # RESTART
    # =====================================

    def restart():

        nonlocal state
        nonlocal player_x
        nonlocal lives, score, distance
        nonlocal speed, scroll, spawn_timer
        nonlocal obstacles

        state = "playing"

        player_x = WIDTH / 2

        lives = 3
        score = 0
        distance = 0.0

        speed = 330.0
        scroll = 0.0
        spawn_timer = 1.2

        obstacles = []

    # =====================================
    # MAIN LOOP
    # =====================================

    try:

        running = True

        while running:

            dt = min(
                clock.tick(FPS) / 1000.0,
                0.05
            )

            # Keyboard events
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:

                        running = False

                    elif (
                        event.key == pygame.K_RETURN
                        and state != "playing"
                    ):

                        restart()

                    elif event.key == pygame.K_r:

                        restart()

                    elif event.key == pygame.K_m:

                        camera_mode = not camera_mode

                    elif event.key == pygame.K_i:

                        reverse = not reverse

                    elif event.key == pygame.K_c:

                        ok = wheel.calibrate()

                        notice = (
                            "STEERING CENTERED"
                            if ok
                            else "SHOW BOTH HANDS FIRST"
                        )

                        notice_until = (
                            time.monotonic() + 2.0
                        )

            # Camera snapshot
            (
                valid,
                camera_steer,
                hands,
                status,
                preview
            ) = wheel.snapshot()

            keys = pygame.key.get_pressed()

            key_steer = (
                int(
                    keys[pygame.K_d]
                    or keys[pygame.K_RIGHT]
                )
                -
                int(
                    keys[pygame.K_a]
                    or keys[pygame.K_LEFT]
                )
            )

            if camera_mode and valid:

                steer = (
                    -camera_steer
                    if reverse
                    else camera_steer
                )

                mode = "CAMERA"

            else:

                steer = key_steer

                mode = (
                    "KEYBOARD BACKUP"
                    if camera_mode
                    else "KEYBOARD"
                )

            # =================================
            # GAME UPDATE
            # =================================

            if state == "playing":

                speed = min(
                    530,
                    330 + distance * 0.012
                )

                scroll += speed * dt

                distance += (
                    speed * dt / 10
                )

                # Steering
                player_x += steer * 355 * dt

                player_x = clamp(
                    player_x,
                    ROAD_LEFT + 40,
                    ROAD_RIGHT - 40
                )

                # Traffic spawning
                spawn_timer -= dt

                if spawn_timer <= 0:

                    lane = random.randrange(3)

                    lane_x = (
                        ROAD_LEFT
                        + (
                            ROAD_RIGHT - ROAD_LEFT
                        ) * (lane + 0.5) / 3
                    )

                    if (
                        not obstacles
                        or obstacles[-1]["y"] > 150
                    ):

                        obstacles.append({
                            "x": lane_x,
                            "y": -75.0,
                            "paint": random.choice([
                                (238, 90, 80),
                                (250, 183, 66),
                                (144, 106, 218),
                                (229, 235, 241)
                            ])
                        })

                    spawn_timer = random.uniform(
                        0.9,
                        1.5
                    )

            # =================================
            # DRAW ROAD AND CARS
            # =================================

            draw_road(
                screen,
                scroll
            )

            player_rect = draw_car(
                screen,
                player_x,
                PLAYER_Y,
                (20, 203, 228),
                player=True
            )

            for enemy in obstacles[:]:

                if state == "playing":

                    enemy["y"] += (
                        speed * dt * 0.88
                    )

                enemy_rect = draw_car(
                    screen,
                    enemy["x"],
                    enemy["y"],
                    enemy["paint"]
                )

                # Collision
                if (
                    state == "playing"
                    and enemy_rect.colliderect(
                        player_rect
                    )
                ):

                    obstacles.remove(enemy)

                    lives -= 1

                    if lives <= 0:
                        state = "gameover"

                elif enemy["y"] > HEIGHT + 90:

                    obstacles.remove(enemy)

                    if state == "playing":
                        score += 1

            # =================================
            # MENU / GAME OVER
            # =================================

            if state != "playing":

                overlay = pygame.Surface(
                    (WIDTH, HEIGHT),
                    pygame.SRCALPHA
                )

                overlay.fill(
                    (7, 13, 23, 169)
                )

                screen.blit(
                    overlay,
                    (0, 0)
                )

                pygame.draw.rect(
                    screen,
                    (21, 38, 56),
                    (140, 260, 620, 250),
                    border_radius=22
                )

                heading = (
                    "STEERING RACE 2D"
                    if state == "menu"
                    else "RACE OVER"
                )

                draw_text(
                    screen,
                    font_big,
                    heading,
                    WIDTH // 2,
                    314,
                    CYAN,
                    center=True
                )

                message = (
                    "Hold TWO hands like a steering wheel"
                    if state == "menu"
                    else (
                        f"Passed: {score}  |  "
                        f"Distance: {int(distance)} m"
                    )
                )

                draw_text(
                    screen,
                    font_small,
                    message,
                    WIDTH // 2,
                    382,
                    WHITE,
                    center=True
                )

                draw_text(
                    screen,
                    font_mid,
                    "PRESS ENTER TO START",
                    WIDTH // 2,
                    454,
                    YELLOW,
                    center=True
                )

            # =================================
            # HUD
            # =================================

            pygame.draw.rect(
                screen,
                (9, 23, 36),
                (18, 16, 292, 147),
                border_radius=16
            )

            draw_text(
                screen,
                font_mid,
                "STEERING RACE",
                32,
                24,
                CYAN
            )

            draw_text(
                screen,
                font_small,
                f"SPEED {int(speed / 5):03d} km/h",
                33,
                65
            )

            draw_text(
                screen,
                font_small,
                f"DISTANCE {int(distance)} m",
                33,
                94
            )

            draw_text(
                screen,
                font_small,
                f"PASS {score}   LIVES {lives}",
                33,
                123,
                YELLOW
            )

            # =================================
            # CAMERA PREVIEW
            # =================================

            px = WIDTH - 305
            py = 16

            pygame.draw.rect(
                screen,
                (9, 23, 36),
                (px, py, 293, 216),
                border_radius=14
            )

            if preview is not None:

                try:

                    preview_surface = (
                        pygame.image.frombuffer(
                            preview,
                            (PREVIEW_W, PREVIEW_H),
                            "RGB"
                        )
                    )

                    screen.blit(
                        preview_surface,
                        (px + 10, py + 9)
                    )

                except (ValueError, pygame.error):
                    pass

            else:

                draw_text(
                    screen,
                    font_small,
                    "CAMERA STARTING...",
                    px + 20,
                    py + 55
                )

            draw_text(
                screen,
                font_tiny,
                f"{mode}  |  {hands}/2 HANDS",
                px + 11,
                py + 167,
                CYAN if valid else YELLOW
            )

            draw_text(
                screen,
                font_tiny,
                status[:32],
                px + 11,
                py + 190
            )

            # =================================
            # STEERING INDICATOR
            # =================================

            pygame.draw.rect(
                screen,
                DARK,
                (
                    WIDTH // 2 - 115,
                    HEIGHT - 59,
                    230,
                    23
                ),
                border_radius=12
            )

            pygame.draw.line(
                screen,
                WHITE,
                (WIDTH // 2, HEIGHT - 58),
                (WIDTH // 2, HEIGHT - 38),
                2
            )

            pygame.draw.circle(
                screen,
                CYAN,
                (
                    WIDTH // 2
                    + int(steer * 101),
                    HEIGHT - 47
                ),
                10
            )

            draw_text(
                screen,
                font_tiny,
                (
                    "A/D STEER   M MODE   "
                    "C CENTER   I REVERSE   "
                    "R RESTART   ESC EXIT"
                ),
                18,
                HEIGHT - 27
            )

            if time.monotonic() < notice_until:

                draw_text(
                    screen,
                    font_mid,
                    notice,
                    WIDTH // 2,
                    229,
                    YELLOW,
                    center=True
                )

            pygame.display.flip()

    finally:

        wheel.close()
        pygame.quit()


# =========================================
# START APPLICATION
# =========================================

if __name__ == "__main__":
    main()
