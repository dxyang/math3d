import pytest

import math
import numpy as np

from vec3 import Vec3


def test_default_init():
    v = Vec3()
    assert v.x == 0.0
    assert v.y == 0
    assert v.z == 0


def test_getters():
    v = Vec3(0.0, np.pi, 100)
    assert v.x == 0.0
    assert v.y == np.pi
    assert v.z == 100


def test_setters():
    v = Vec3()
    v.x = 10
    v.y = 40.0
    v.z = 22 / 3.0
    assert v.x == 10
    assert v.y == 40
    assert v.z == 22 / 3.0


def test_eq():
    v1 = Vec3(0, 1, 2)
    v2 = Vec3(0.0, 1.0, 2.0)

    assert v1 == v2


def test_ne():
    v1 = Vec3(5, 7, 8)
    v2 = Vec3(0.0, 1.0, 2.0)

    assert v1 != v2


def test_neg():
    v1 = Vec3(1, 2, 3)
    v2 = -v1
    v_true = Vec3(-1, -2, -3)

    assert v2 == v_true


def test_add():
    v1 = Vec3(1, 1, 1)
    v2 = Vec3(2, 2, 2)

    v3 = v1 + v2

    assert isinstance(v3, Vec3)

    v_true = Vec3(3, 3, 3)

    assert v3 == v_true

    v1 += v2

    assert v1 == v_true


def test_sub():
    v1 = Vec3(1, 1, 1)

    v2 = Vec3(4, 4, 4)

    v3 = v1 - v2

    assert isinstance(v3, Vec3)

    v_true = Vec3(-3, -3, -3)

    assert v3 == v_true

    v1 -= v2

    assert v1 == v_true


def test_mul():
    v1 = Vec3(1, 2, 3)
    v2 = v1 * -4
    v3 = -4 * v1

    assert isinstance(v2, Vec3)
    assert isinstance(v3, Vec3)
    assert v2 == v3

    v_true = Vec3(-4, -8, -12)

    assert v2 == v_true

    v1 *= -4

    assert v1 == v_true


def test_div():
    v1 = Vec3(1, 2, 3)
    v2 = v1 / -4

    assert isinstance(v2, Vec3)

    v_true = Vec3(-0.25, -0.5, -0.75)

    assert v2 == v_true

    v1 /= -4

    assert v1 == v_true


def test_dot():
    v1 = Vec3(0, 1, 2)
    v2 = Vec3(3, 4, 5)

    res = v1.dot(v2)

    assert res == 14


def test_cross():
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(4.0, 5.0, 6.0)

    v3 = v1.cross(v2)

    v_true = Vec3(-3, 6, -3)

    assert v3 == v_true

    v4 = v2.cross(v1)

    v_true = Vec3(3, -6, 3)

    assert v4 == v_true


def test_length():
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(3, 0, 4)

    assert v1.length() == math.sqrt(14)
    assert v1.length_squared() == 14
    assert v2.length() == 5
    assert v2.length_squared() == 25


def test_distance():
    v1 = Vec3(1.0, 2.0, 3.0)
    v2 = Vec3(3, 0, 4)

    assert v1.distance(v2) == 3
    assert v1.distance_squared(v2) == 9
    assert v2.distance(v1) == 3
    assert v2.distance_squared(v1) == 9
    assert v2.distance(v2) == 0


def test_normalize():
    v1 = Vec3(math.sqrt(3) / 3, -math.sqrt(3) / 3, math.sqrt(3) / 3)
    assert v1.is_normalized()

    v2 = Vec3(3, 0, 4)
    v3 = v2.normalized()
    assert not v2.is_normalized()

    v2.normalize()
    assert v2.is_normalized()

    assert v2 == v3

    v_true = Vec3(0.6, 0, 0.8)
    assert v_true == v3


def test_project():
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(3, -4, -5)

    v3 = v1.project_to(v2)

    v_true = v2.normalized() * (-2 * math.sqrt(2))

    assert v3 == v_true

    v4 = v1.project_to_plane(v2)

    v_true = v1 - v3

    assert v4 == v_true


def test_finite():
    v1 = Vec3(10000000000, 210000000000, 310000000000)
    assert v1.is_finite()

    v2 = Vec3(np.nan, 210000000000, 310000000000)
    assert not v2.is_finite()
    assert v2.is_nan()

    v3 = Vec3(np.inf, 210000000000, 310000000000)
    assert not v3.is_finite()
    assert v3.is_nan()

    v4 = Vec3(0.0, 210000000000, np.NINF)
    assert not v4.is_finite()
    assert v4.is_nan()


def test_from_numpy():
    a = np.array([1, 2, 3])
    v1 = Vec3.from_numpy(a)

    assert a[0] == v1.x
    assert a[1] == v1.y
    assert a[2] == v1.z


def test_to_numpy():
    a = np.array([1, 2, 3])
    v1 = Vec3.from_numpy(a)
    b = v1.to_numpy()

    assert (a == b).all()


def test_quick_constructors():
    v1 = Vec3.zero()
    assert v1 == Vec3(0, 0, 0)
    v2 = Vec3.x_axis()
    assert v2 == Vec3(1, 0, 0)
    v3 = Vec3.y_axis()
    assert v3 == Vec3(0, 1, 0)
    v4 = Vec3.z_axis()
    assert v4 == Vec3(0, 0, 1)
