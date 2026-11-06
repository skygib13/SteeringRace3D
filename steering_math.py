"""Pure-Python steering geometry.  Angle is in radians, positive = right."""
import math


def angle_between_hands(left_xy, right_xy):
    lx, ly = left_xy
    rx, ry = right_xy
    # Screen coordinates: x increases rightward, y downward.
    # Lower right hand -> clockwise imaginary wheel turn -> steer right.
    return math.atan2(ry - ly, rx - lx)


def steering_from_angle(angle, zero_angle=0.0, max_angle_degrees=34.0,
                        deadzone_degrees=3.5):
    delta = math.atan2(math.sin(angle - zero_angle),
                       math.cos(angle - zero_angle))
    dead = math.radians(deadzone_degrees)
    maximum = math.radians(max_angle_degrees)
    if abs(delta) <= dead:
        return 0.0
    strength = (abs(delta) - dead) / max(0.001, (maximum - dead))
    return math.copysign(min(1.0, strength), delta)
