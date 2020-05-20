import pytest

import math
import numpy as np

from quat import Quat
from transform import Transform
from vec3 import Vec3


def test_init_and_eq_and_ne():
    t1 = Transform()
    assert t1.rotation == Quat.identity()
    assert t1.translation == Vec3.zero()

    rot = Quat.from_rotation_on_axis(2, np.pi)
    trans = Vec3(1, 2, 3)
    t2 = Transform(rot, trans)
    assert t2.rotation == rot
    assert t2.translation == trans

    t3 = Transform.identity()

    assert t1 == t3
    assert t1 != t2
    assert t3 != t2


def test_mul():
    T_a_b = Transform(Quat.from_rotation_on_axis(2, np.pi / 2), Vec3(1, 1, 1))
    T_b_c = Transform(Quat.from_rotation_on_axis(0, np.pi / 2), Vec3(-1, -2, -3))
    c_p = Vec3(2, 3, 4)

    T_a_c = T_a_b * T_b_c

    assert isinstance(T_a_c, Transform)

    a_p = T_a_c * c_p
    assert isinstance(a_p, Vec3)

    # fmt: off
    np_T_a_b = np.array([
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ])
    np_T_b_c = np.array([
        [1, 0, 0, -1],
        [0, 0, -1, -2],
        [0, 1, 0, -3],
        [0, 0, 0, 1],
    ])
    np_c_p = np.array([
        [2],
        [3],
        [4],
        [1],
    ])
    # fmt: on

    npgt_T_a_c = np_T_a_b @ np_T_b_c
    npgt_a_p = np.squeeze(npgt_T_a_c @ np_c_p)[0:3]
    np_T_a_c = T_a_c.to_mat4_numpy()
    np_a_p = a_p.to_numpy()


    assert np.allclose(np_T_a_c, npgt_T_a_c)
    assert np.allclose(npgt_a_p, np_a_p)


def test_numpy_interop():
    rot = Quat.from_rotation_on_axis(2, np.pi / 2)
    trans = Vec3(1, 1, 1)

    # fmt: off
    np_T = np.array([
        [0, -1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ])
    # fmt: on

    T_a_b = Transform(rot, trans)
    T_a_b_from_mat4 = Transform.from_mat4_numpy(np_T)
    T_a_b_from_quatvec3 = Transform.from_quat_vec3_numpy(
        np.concatenate((rot.to_xyzw_numpy(), trans.to_numpy()))
    )

    assert T_a_b == T_a_b_from_mat4
    assert T_a_b == T_a_b_from_quatvec3
    assert T_a_b_from_quatvec3 == T_a_b_from_mat4

    np_mat4_T = T_a_b.to_mat4_numpy()
    np_quatvec3_T = T_a_b.to_quat_vec3_numpy()

    assert np.allclose(
        np_quatvec3_T, np.concatenate((rot.to_xyzw_numpy(), trans.to_numpy()))
    )
    assert np.allclose(np_mat4_T, np_T)


def test_rotate_vector():
    v = Vec3(1, 1, 0)

    q1 = Quat.from_rotation_on_axis(0, np.pi / 3)
    q2 = Quat.from_rotation_on_axis(1, np.pi / 3)
    q3 = Quat.from_rotation_on_axis(2, np.pi / 3)

    t1 = Transform(q1, Vec3.zero())
    t2 = Transform(q2, Vec3.zero())
    t3 = Transform(q3, Vec3.zero())

    # forward rotation
    v1 = t1.rotate(v)
    v2 = t2.rotate(v)
    v3 = t3.rotate(v)

    theta1 = (60) * np.pi / 180.0
    v1_true = Vec3(1, math.cos(theta1), math.sin(theta1))
    v2_true = Vec3(math.cos(theta1), 1, -math.sin(theta1))

    theta2 = (45 + 60) * np.pi / 180.0
    v3_true = math.sqrt(2) * Vec3(math.cos(theta2), math.sin(theta2), 0)

    assert v1 == v1_true
    assert v2 == v2_true
    assert v3 == v3_true

    # inverse rotate
    v1_rev = t1.inverse_rotate(v1_true)
    v2_rev = t2.inverse_rotate(v2_true)
    v3_rev = t3.inverse_rotate(v3_true)

    assert v == v1_rev
    assert v == v2_rev
    assert v == v3_rev


def test_translate_vector():
    v = Vec3(1, 1, 0)

    t = Transform(Quat.identity(), Vec3(4, 5, 6))

    res = t.translate(v)

    assert res == Vec3(5, 6, 6)


def test_transform_vector():
    v = Vec3(1, 1, 0)

    q1 = Quat.from_rotation_on_axis(0, np.pi / 3)
    q2 = Quat.from_rotation_on_axis(1, np.pi / 3)
    q3 = Quat.from_rotation_on_axis(2, np.pi / 3)

    t1 = Transform(q1, Vec3(1, 1, -1))
    t2 = Transform(q2, Vec3(4, -3, -1))
    t3 = Transform(q3, Vec3(-1, 1, -1))

    # forward rotation
    v1 = t1.transform(v)
    v2 = t2.transform(v)
    v3 = t3.transform(v)

    theta1 = (60) * np.pi / 180.0
    v1_true = Vec3(1 + 1, math.cos(theta1) + 1, math.sin(theta1) - 1)
    v2_true = Vec3(math.cos(theta1) + 4, 1 - 3, -math.sin(theta1) - 1)

    theta2 = (45 + 60) * np.pi / 180.0
    v3_true = Vec3(
        math.sqrt(2) * math.cos(theta2) - 1, math.sqrt(2) * math.sin(theta2) + 1, 0 - 1
    )

    assert v1 == v1_true
    assert v2 == v2_true
    assert v3 == v3_true

    # inverse rotate
    v1_rev = t1.inverse_transform(v1_true)
    v2_rev = t2.inverse_transform(v2_true)
    v3_rev = t3.inverse_transform(v3_true)

    assert v == v1_rev
    assert v == v2_rev
    assert v == v3_rev


def test_inverse():
    t = Transform(Quat.from_axis_angle(Vec3(1, 1, 1), np.pi / 2), Vec3(4, 5, 6))
    t_inv = t.inverse()

    t_identity_one = t * t_inv
    t_identity_two = t_inv * t

    np_identity = np.diag((1, 1, 1, 1))
    t_fromnp = Transform.from_mat4_numpy(np_identity)

    assert np.allclose(t_identity_one.to_mat4_numpy(), np_identity)
    assert np.allclose(t_identity_two.to_mat4_numpy(), np_identity)
    assert np.allclose(t_identity_one.to_mat4_numpy(), t_identity_two.to_mat4_numpy())
    assert t_identity_one == t_fromnp
    assert t_identity_two == t_fromnp
    assert t_identity_one == t_identity_two


def test_normalize():
    q1 = Quat(0.4619398, 0.1913417, 0.4619398, 0.7325378)  # 45 around X then Y then Z
    t1 = Transform(q1, Vec3.zero())

    assert t1.is_normalized()

    q2 = Quat(3, 0, 4, 7)
    t2 = Transform(q2, Vec3.zero())
    assert not t2.is_normalized()

    t2.normalize()
    assert t2.is_normalized()


def test_finte():
    q1 = Quat(10000000000, 210000000000, 310000000000, -2147483647)
    q2 = Quat(np.nan, 210000000000, 310000000000, -2147483647)
    q3 = Quat(np.inf, 210000000000, 310000000000, -2147483647)
    q4 = Quat(0.0, 210000000000, np.NINF, -2147483647)

    v1 = Vec3(10000000000, 210000000000, 310000000000)
    v2 = Vec3(np.nan, 210000000000, 310000000000)
    v3 = Vec3(np.inf, 210000000000, 310000000000)
    v4 = Vec3(0.0, 210000000000, np.NINF)

    t1 = Transform(q1, v1)
    assert t1.is_finite()
    assert not t1.is_nan()

    t2 = Transform(q1, v2)
    assert not t2.is_finite()
    assert t2.is_nan()

    t3 = Transform(q3, v1)
    assert not t3.is_finite()
    assert t3.is_nan()

    t4 = Transform(q4, v4)
    assert not t4.is_finite()
    assert t4.is_nan()
