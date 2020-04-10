import pytest

import math
import numpy as np

from quat import Quat
from vec3 import Vec3
from _constants import MATH_EPS


def test_default_init():
    q = Quat()
    assert q.x == 0.0
    assert q.y == 0
    assert q.z == 0
    assert q.w == 1.0


def test_getters():
    q = Quat(0.0, np.pi, 100, 1)
    assert q.x == 0.0
    assert q.y == np.pi
    assert q.z == 100
    assert q.w == 1


def test_setters():
    q = Quat()
    q.x = 10
    q.y = 40.0
    q.z = 22 / 3.0
    q.w = np.pi
    assert q.x == 10
    assert q.y == 40
    assert q.z == 22 / 3.0
    assert q.w == np.pi


def test_eq():
    q1 = Quat(0, 1, 2, 3.0)
    q2 = Quat(0.0, 1.0, 2.0, 3)

    assert q1 == q2


def test_ne():
    q1 = Quat(5, 7, 8, 33 / 2.0)
    q2 = Quat(0.0, 1.0, 2.0, 6)

    assert q1 != q2


def test_neg():
    q1 = Quat(0.0, 1.0, 2.0, 6)
    q2 = -q1
    qtrue = Quat(0, -1, -2, -6)

    assert q2 == qtrue


def test_add():
    q1 = Quat(1, 1, 1, 1)
    q2 = Quat(2, 2, 2, 2)

    q3 = q1 + q2

    assert isinstance(q3, Quat)

    q_true = Quat(3, 3, 3, 3)

    assert q3 == q_true

    q1 += q2

    assert q1 == q_true


def test_sub():
    q1 = Quat(1, 1, 1, 1)

    q2 = Quat(4, 4, 4, 4)

    q3 = q1 - q2

    assert isinstance(q3, Quat)

    q_true = Quat(-3, -3, -3, -3)

    assert q3 == q_true

    q1 -= q2

    assert q1 == q_true


def test_mul():
    q1 = Quat(1, 2, 3, -5)
    q2 = q1 * -4
    q3 = -4 * q1

    assert isinstance(q2, Quat)
    assert isinstance(q3, Quat)
    assert q2 == q3

    q_true = Quat(-4, -8, -12, 20)

    assert q2 == q_true

    q1 *= -4

    assert q1 == q_true


def test_div():
    q1 = Quat(1, 2, 3, -5)
    q2 = q1 / -4

    assert isinstance(q2, Quat)

    q_true = Quat(-0.25, -0.5, -0.75, 1.25)

    assert q2 == q_true

    q1 /= -4

    assert q1 == q_true


def test_dot():
    q1 = Quat(0, 1, 2, 3)
    q2 = Quat(3, 4, 5, 6)

    res = q1.dot(q2)

    assert res == 32


def test_length():
    q1 = Quat(1.0, 2.0, 3.0, 4)
    q2 = Quat(3, 0, 4, 5)

    assert q1.length() == math.sqrt(30)
    assert q1.length_squared() == 30
    assert q2.length() == math.sqrt(50)
    assert q2.length_squared() == 50


def test_distance():
    q1 = Quat(1.0, 2.0, 3.0, 4)
    q2 = Quat(3, 0, 4, 5)

    assert q1.distance(q2) == math.sqrt(10)
    assert q1.distance_squared(q2) == 10
    assert q2.distance(q1) == math.sqrt(10)
    assert q2.distance_squared(q1) == 10
    assert q2.distance(q2) == 0


def test_angle():
    q1 = Quat(0.7071068, 0, 0, 0.7071068)  # 90 degrees around x axis
    q2 = Quat(0, 0.3826834, 0, 0.9238795)  # 45 around y axis

    assert abs(q1.angle() - np.pi / 2) < MATH_EPS
    assert abs(q2.angle() == np.pi / 4) < MATH_EPS

    res = q1.angle(q2)
    assert abs(res - 1.7177715322714415) < MATH_EPS


def test_normalize():
    q1 = Quat(0.4619398, 0.1913417, 0.4619398, 0.7325378)  # 45 around X then Y then Z
    assert q1.is_normalized()

    q2 = Quat(3, 0, 4, 7)
    assert not q2.is_normalized()

    q2.normalize()
    assert q2.is_normalized()


def test_inverse():
    q1 = Quat(0.4619398, 0.1913417, 0.4619398, 0.7325378)  # 45 around X then Y then Z
    q2 = q1.inverse()

    q_true = Quat(-0.4619398, -0.1913417, -0.4619398, 0.7325378)

    assert q2 == q_true

    q1.invert()
    assert q1 == q_true


def test_finite():
    q1 = Quat(10000000000, 210000000000, 310000000000, -2147483647)
    assert q1.is_finite()

    q2 = Quat(np.nan, 210000000000, 310000000000, -2147483647)
    assert not q2.is_finite()
    assert q2.is_nan()

    q3 = Quat(np.inf, 210000000000, 310000000000, -2147483647)
    assert not q3.is_finite()
    assert q3.is_nan()

    q4 = Quat(0.0, 210000000000, np.NINF, -2147483647)
    assert not q4.is_finite()
    assert q4.is_nan()


def test_rotate_vec():
    v = Vec3(1, 1, 0)

    q1 = Quat.from_rotation_on_axis(0, np.pi / 3)
    q2 = Quat.from_rotation_on_axis(1, np.pi / 3)
    q3 = Quat.from_rotation_on_axis(2, np.pi / 3)

    # forward rotation
    v1 = q1.rotate(v)
    v2 = q2.rotate(v)
    v3 = q3.rotate(v)

    theta1 = (60) * np.pi / 180.0
    v1_true = Vec3(1, math.cos(theta1), math.sin(theta1))
    v2_true = Vec3(math.cos(theta1), 1, -math.sin(theta1))

    theta2 = (45 + 60) * np.pi / 180.0
    v3_true = math.sqrt(2) * Vec3(math.cos(theta2), math.sin(theta2), 0)

    assert v1 == v1_true
    assert v2 == v2_true
    assert v3 == v3_true

    # inverse rotate
    v1_rev = q1.inverse_rotate(v1_true)
    v2_rev = q2.inverse_rotate(v2_true)
    v3_rev = q3.inverse_rotate(v3_true)

    assert v == v1_rev
    assert v == v2_rev
    assert v == v3_rev


def test_euler_angles():
    # Generated from - https://www.andre-gaschler.com/rotationconverter/
    # 30 degrees around X, 9 around Y, 153 around Z
    q1 = Quat(0.1339256, -0.2332002, 0.9410824, 0.2050502)

    x, y, z = q1.get_euler_angles(angle_order=[0, 1, 2])
    assert abs(x - np.pi / 6) < MATH_EPS
    assert abs(y - np.pi / 20) < MATH_EPS
    assert abs(z - 153 * np.pi / 180) < MATH_EPS

    x, z, y = q1.get_euler_angles(angle_order=[0, 2, 1])
    x_true, y_true, z_true = -2.6975331, 2.9656712, 0.4649757
    assert abs(x - x_true) < MATH_EPS
    assert abs(y - y_true) < MATH_EPS
    assert abs(z - z_true) < MATH_EPS

    y, x, z = q1.get_euler_angles(angle_order=[1, 0, 2])
    x_true, y_true, z_true = 0.5165051, 0.1808875, 2.7604268
    assert abs(x - x_true) < MATH_EPS
    assert abs(y - y_true) < MATH_EPS
    assert abs(z - z_true) < MATH_EPS

    yaw, pitch, roll = q1.get_yaw_pitch_roll()
    assert abs(pitch - x_true) < MATH_EPS
    assert abs(yaw - y_true) < MATH_EPS
    assert abs(roll - z_true) < MATH_EPS

    y, z, x = q1.get_euler_angles(angle_order=[1, 2, 0])
    x_true, y_true, z_true = 2.5925117, -2.7653147, 0.3293999
    assert abs(x - x_true) < MATH_EPS
    assert abs(y - y_true) < MATH_EPS
    assert abs(z - z_true) < MATH_EPS

    z, x, y = q1.get_euler_angles(angle_order=[2, 0, 1])
    x_true, y_true, z_true = -0.3941227, -0.3860976, 2.6345058
    assert abs(x - x_true) < MATH_EPS
    assert abs(y - y_true) < MATH_EPS
    assert abs(z - z_true) < MATH_EPS

    z, y, x = q1.get_euler_angles(angle_order=[2, 1, 0])
    x_true, y_true, z_true = -0.4219639, -0.3551227, 2.7893517
    assert abs(x - x_true) < MATH_EPS
    assert abs(y - y_true) < MATH_EPS
    assert abs(z - z_true) < MATH_EPS


def test_rotation_vector():
    # Generated from - https://www.andre-gaschler.com/rotationconverter/
    # 36 degrees around axis (1, 2, 3)
    q1 = Quat(0.0825883, 0.1651765, 0.2477648, 0.9510565)
    v = q1.to_rotation_vector()

    theta = v.length()
    axis = v.normalized()

    theta_true = 36 * np.pi / 180
    axis_true = Vec3(1 / math.sqrt(14), 2 / math.sqrt(14), 3 / math.sqrt(14))

    assert abs(theta - theta_true) < MATH_EPS
    assert axis == axis_true

    rotation_vec = theta_true * axis_true
    q2 = Quat.from_rotation_vector(rotation_vec)
    assert q1 == q2

    q3 = Quat.from_axis_angle(axis_true, theta_true)
    assert q1 == q3


def test_slerp():
    q1 = Quat.from_rotation_on_axis(2, -np.pi / 4)
    q2 = Quat.from_rotation_on_axis(2, np.pi / 4)

    q3 = q1.slerp(q2, 0.5)
    q4 = q2.slerp(q1, 0.5)

    q_true = Quat.identity()

    assert q3 == q_true
    assert q4 == q_true

    q5 = q1.slerp(q2, 0.9)
    q_true = Quat.from_rotation_on_axis(2, (45 - 9) * np.pi / 180)

    assert q5 == q_true


def test_identity_constructors():
    q1 = Quat.identity()
    assert q1 == Quat(0, 0, 0, 1)


def test_major_axis_constructors():
    q1 = Quat.from_rotation_on_axis(0, np.pi / 4)
    q2 = Quat.from_rotation_on_axis(1, np.pi / 3)
    q3 = Quat.from_rotation_on_axis(2, np.pi / 2)

    q1_true = Quat(0.3826834, 0, 0, 0.9238795)
    q2_true = Quat(0, 0.5, 0, 0.8660254)
    q3_true = Quat(0, 0, 0.7071068, 0.7071068)

    assert q1 == q1_true
    assert q2 == q2_true
    assert q3 == q3_true


def test_mat3_numpy():
    q1_true = Quat()
    m1_true = np.diag((1, 1, 1))

    q1 = Quat.from_mat3_numpy(m1_true)
    m1 = q1_true.to_mat3_numpy()

    assert q1 == q1_true
    assert np.allclose(m1, m1_true)

    q2_true = Quat(0.6374259, 0.7264568, 0.204462, -0.1553838)
    m2_true = np.array(
        [
            [-0.1390883, 0.9896649, 0.0348995],
            [0.8625845, 0.1037672, 0.4951569],
            [0.4864179, 0.0989743, -0.8681024],
        ]
    )

    q2 = Quat.from_mat3_numpy(m2_true)
    m2 = q2_true.to_mat3_numpy()

    assert q2 == q2_true
    assert np.allclose(m2, m2_true)


def test_xyzw_numpy():
    a = np.array([1, 2, 3, 4])
    q1 = Quat.from_xyzw_numpy(a)
    b = q1.to_xyzw_numpy()

    assert a[0] == q1.x
    assert a[1] == q1.y
    assert a[2] == q1.z
    assert a[3] == q1.w
    assert np.allclose(a, b)


def test_from_numpy():
    m = np.array(
        [
            [-0.1390883, 0.9896649, 0.0348995],
            [0.8625845, 0.1037672, 0.4951569],
            [0.4864179, 0.0989743, -0.8681024],
        ]
    )
    q = np.array([1, 2, 3, 4])

    qm_true = Quat(0.6374259, 0.7264568, 0.204462, -0.1553838)
    qq_true = Quat(1, 2, 3, 4)

    qm = Quat.from_numpy(m)
    qq = Quat.from_numpy(q)

    assert qm == qm_true
    assert qq == qq_true
