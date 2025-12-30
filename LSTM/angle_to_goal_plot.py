import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# stadium (105 x 68), attacking direction: left -> right
STADIUM_X, STADIUM_Y = 105.0, 68.0
CENTER_Y = STADIUM_Y / 2.0  # 34.0

# goal (right side)
GOAL_X = STADIUM_X
GOAL_POST_HALF = 3.66
GOAL_Y_L = CENTER_Y - GOAL_POST_HALF  # 30.34
GOAL_Y_R = CENTER_Y + GOAL_POST_HALF  # 37.66

def plot_goal_angles(sx, sy, arc_radius=8.0):
    dxg = GOAL_X - sx

    # angles to posts
    alpha_L = np.arctan2(GOAL_Y_L - sy, dxg)  # rad
    alpha_R = np.arctan2(GOAL_Y_R - sy, dxg)  # rad

    # mid + view
    theta_mid = (alpha_L + alpha_R) / 2.0
    theta_view = np.abs(alpha_R - alpha_L)

    # degrees for plotting arc
    aL = np.degrees(alpha_L)
    aR = np.degrees(alpha_R)
    a1, a2 = (min(aL, aR), max(aL, aR))
    am = np.degrees(theta_mid)

    fig, ax = plt.subplots(figsize=(10, 6))

    # pitch boundary
    ax.plot([0, STADIUM_X, STADIUM_X, 0, 0],
            [0, 0, STADIUM_Y, STADIUM_Y, 0])

    # goal segment
    ax.plot([GOAL_X, GOAL_X], [GOAL_Y_L, GOAL_Y_R], linewidth=3)

    # point (ball/event start)
    ax.scatter([sx], [sy], s=60)
    ax.text(sx + 1.0, sy + 1.0, f"({sx:.1f}, {sy:.1f})")

    # rays to posts
    ax.plot([sx, GOAL_X], [sy, GOAL_Y_L])
    ax.plot([sx, GOAL_X], [sy, GOAL_Y_R])

    # ray to mid direction (use dx=arc_radius for visual length)
    ax.plot([sx, sx + arc_radius * np.cos(theta_mid)],
            [sy, sy + arc_radius * np.sin(theta_mid)])

    # arc showing theta_view (between post directions)
    arc = Arc((sx, sy), width=2*arc_radius, height=2*arc_radius,
              angle=0.0, theta1=a1, theta2=a2)
    ax.add_patch(arc)

    # labels
    ax.text(sx + arc_radius * np.cos(theta_mid),
            sy + arc_radius * np.sin(theta_mid),
            f"theta_mid={am:.1f}°")

    ax.text(sx + arc_radius * np.cos(theta_mid) + 1.0,
            sy + arc_radius * np.sin(theta_mid) - 2.0,
            f"theta_view={np.degrees(theta_view):.1f}°")

    ax.text(sx + arc_radius * np.cos(np.radians(a1)),
            sy + arc_radius * np.sin(np.radians(a1)),
            f"alpha_L={aL:.1f}°")

    ax.text(sx + arc_radius * np.cos(np.radians(a2)),
            sy + arc_radius * np.sin(np.radians(a2)),
            f"alpha_R={aR:.1f}°")

    ax.set_xlim(-2, STADIUM_X + 5)
    ax.set_ylim(-2, STADIUM_Y + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Angles to Goal Posts: alpha_L, alpha_R, theta_mid, theta_view")
    plt.show()

# 예시 1: 중앙 근처
plot_goal_angles(90, 34)

# 예시 2: 측면에서
plot_goal_angles(90, 10)

# 예시 3: 더 먼 지점
plot_goal_angles(60, 34)
