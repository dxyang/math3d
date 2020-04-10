from typing import List, Union

import math
import numpy as np

from _constants import MATH_EPS, MATH_SINGULARITYRADIUS
from vec3 import Vec3


class Quat:
    """
    Quat represents a quaternion class used for rotations.

    Quaternion multiplications are done in right-to-left order,
    to match the behavior of matrices.
    """

    def __init__(self, _x: float = 0, _y: float = 0, _z: float = 0, _w: float = 1):
        self.x = _x
        self.y = _y
        self.z = _z
        self.w = _w

    """
    python overrides
    """

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        elif key == 3:
            return self.w
        else:
            assert False  # quat only has 4 things

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        elif key == 3:
            self.w = value
        else:
            assert False  # quat only has 4 things

    def __str__(self):
        return "x: {}, y: {}, z:{}, w: {}".format(self.x, self.y, self.z, self.w)

    def __repr__(self):
        return 'Quat({}, {}, {}, {})'.format(self.x, self.y, self.z, self.w)

    def __add__(self, rhs: "Quat"):
        return Quat(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)

    def __iadd__(self, rhs: "Quat"):
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z
        self.w += rhs.w
        return self

    def __sub__(self, rhs: "Quat"):
        return Quat(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)

    def __isub__(self, rhs: "Quat"):
        self.x -= rhs.x
        self.y -= rhs.y
        self.z -= rhs.z
        self.w -= rhs.w
        return self

    def __mul__(self, rhs: "Quat"):
        t = type(rhs)
        if t is Quat:
            return Quat(
                self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
                self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
                self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            )
        elif (t is int) or (t is float):
            return Quat(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
        else:
            assert False

    def __rmul__(self, lhs: Union[int, float]):
        t = type(lhs)
        if (t is int) or (t is float):
            return Quat(self.x * lhs, self.y * lhs, self.z * lhs, self.w * lhs)
        else:
            assert False

    def __imul__(self, rhs: Union[int, float, "Quat"]):
        self = self * rhs
        return self

    def __truediv__(self, scalar: float):
        return Quat(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def __idiv__(self, scalar: float):
        self.x /= scalar
        self.y /= scalar
        self.z /= scalar
        self.w /= scalar
        return self

    def __neg__(self):
        return Quat(-self.x, -self.y, -self.z, -self.w)

    def __eq__(self, rhs: "Quat"):
        return (
            abs(self.x - rhs.x) <= MATH_EPS
            and abs(self.y - rhs.y) <= MATH_EPS
            and abs(self.z - rhs.z) <= MATH_EPS
            and abs(self.w - rhs.w) <= MATH_EPS
        )

    def __ne__(self, rhs: "Quat"):
        return (
            abs(self.x - rhs.x) > MATH_EPS
            or abs(self.y - rhs.y) > MATH_EPS
            or abs(self.z - rhs.z) > MATH_EPS
            or abs(self.w - rhs.w) > MATH_EPS
        )

    """
    useful functions
    """

    def dot(self, rhs: "Quat"):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def length(self):
        return math.sqrt(self.length_squared())

    def distance_squared(self, b: "Quat"):
        delta = b - self
        return delta.length_squared()

    def distance(self, b: "Quat"):
        delta = b - self
        return delta.length()

    def angle(self, b: "Quat" = None):
        t = type(b)
        if t is type(None):
            # Quat is rotation around axis xyz by angle w
            return 2 * math.acos(abs(self.w))
        elif t is Quat:
            # Angle between two Quats
            return 2 * math.acos(abs(self.dot(b)))

    def is_normalized(self):
        return abs(self.length_squared() - 1) <= MATH_EPS

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length
        self.w /= length

    def normalized(self):
        return self / self.length()

    def inverse(self):
        return Quat(-self.x, -self.y, -self.z, self.w)

    def invert(self):
        self.x *= -1
        self.y *= -1
        self.z *= -1

    def is_nan(self):
        return not np.isfinite(self.x + self.y + self.z + self.w)

    def is_finite(self):
        return np.isfinite(self.x + self.y + self.z + self.w)

    # transforms vector as if this quat was a matrix rotation
    def rotate(self, v: Vec3):
        # Standard formula: q(t) * V * q(t)^-1
        assert self.is_normalized()  # quat math bug likely

        # rv = q * (v,0) * q'
        # Same as rv = v + real * cross(imag,v)*2 + cross(imag, cross(imag,v)*2);

        # uv = 2 * Imag().Cross(v);
        uvx = 2 * (self.y * v.z - self.z * v.y)
        uvy = 2 * (self.z * v.x - self.x * v.z)
        uvz = 2 * (self.x * v.y - self.y * v.x)

        # return v + Real()*uv + Imag().Cross(uv);
        return Vec3(
            v.x + self.w * uvx + self.y * uvz - self.z * uvy,
            v.y + self.w * uvy + self.z * uvx - self.x * uvz,
            v.z + self.w * uvz + self.x * uvy - self.y * uvx,
        )

    # rotation by inverse of this quat
    def inverse_rotate(self, v: Vec3):
        assert self.is_normalized()  # quat math bug likely

        # rv = q' * (v,0) * q
        # Same as rv = v + real * cross(-imag,v)*2 + cross(-imag, cross(-imag,v)*2);
        #      or rv = v - real * cross(imag,v)*2 + cross(imag, cross(imag,v)*2);

        # uv = 2 * Imag().Cross(v);
        uvx = 2 * (self.y * v.z - self.z * v.y)
        uvy = 2 * (self.z * v.x - self.x * v.z)
        uvz = 2 * (self.x * v.y - self.y * v.x)

        # return v - Real()*uv + Imag().Cross(uv);
        return Vec3(
            v.x - self.w * uvx + self.y * uvz - self.z * uvy,
            v.y - self.w * uvy + self.z * uvx - self.x * uvz,
            v.z - self.w * uvz + self.x * uvy - self.y * uvx,
        )

    # extract euler angles from quat
    # assumes right handed coordinate system
    # CCW rotations looking in the negative axis direction
    # for order abc, the rotations are applied c, then b, then a
    #    XYZ = [0, 1, 2]
    #    XZY = [0, 2, 1]
    #    YXZ = [1, 0, 2]
    #    YZX = [1, 2, 0]
    #    ZXY = [2, 0, 1]
    #    ZYX = [2, 1, 0]
    def get_euler_angles(self, angle_order: List[int] = [0, 1, 2]):
        assert self.is_normalized()  # quat math bug likely
        assert len(angle_order) == 3

        ao = angle_order
        Q = [self.x, self.y, self.z]

        ww = self.w * self.w
        Q00 = Q[ao[0]] * Q[ao[0]]
        Q11 = Q[ao[1]] * Q[ao[1]]
        Q22 = Q[ao[2]] * Q[ao[2]]

        # determine whether even permutation (XYZ, YZX, ZXY)
        psign = -1
        if (((ao[0] + 1) % 3) == ao[1]) and (((ao[1] + 1) % 3) == ao[2]):
            psign = 1

        s2 = psign * 2 * (psign * self.w * Q[ao[1]] + Q[ao[0]] * Q[ao[2]])
        if s2 < -1 + MATH_SINGULARITYRADIUS:
            a = 0
            b = -1 * math.pi / 2.0
            c = math.atan2(
                2 * (psign * Q[ao[0]] * Q[ao[1] + self.w * Q[ao[2]]]),
                ww + Q11 - Q00 - Q22,
            )
        elif s2 > 1 - MATH_SINGULARITYRADIUS:
            a = 0
            b = math.pi / 2.0
            c = math.atan2(
                2 * (psign * Q[ao[0]] * Q[ao[1] + self.w * Q[ao[2]]]),
                ww + Q11 - Q00 - Q22,
            )
        else:
            a = -1 * math.atan2(
                -2 * (self.w * Q[ao[0]] - psign * Q[ao[1]] * Q[ao[2]]),
                ww + Q22 - Q00 - Q11,
            )
            b = math.asin(s2)
            c = math.atan2(
                2 * (self.w * Q[ao[2]] - psign * Q[ao[0]] * Q[ao[1]]),
                ww + Q00 - Q11 - Q22,
            )

        return a, b, c

    # decompose rotation into three rotations:
    # 1) roll around the z axis
    # 2) pitch around the x axis
    # 3) yaw around the y axis
    # these are applied in the order of roll, then pitch, then yaw
    def get_yaw_pitch_roll(self):
        return self.get_euler_angles([1, 0, 2])

    # Convert a quaternion to a rotation vector, also known as
    # Rodrigues vector, AxisAngle vector, SORA vector, exponential map.
    # A rotation vector describes a rotation about an axis:
    # the axis of rotation is the vector normalized,
    # the angle of rotation is the magnitude of the vector.
    def to_rotation_vector(self):
        assert self.is_normalized()  # quat math bug likely

        s = 0
        sinHalfAngle = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if sinHalfAngle > 0:
            cosHalfAngle = self.w
            halfAngle = math.atan2(sinHalfAngle, cosHalfAngle)

            # ensure minimum rotation magnitude
            if cosHalfAngle < 0:
                halfAngle -= np.pi

            s = 2 * halfAngle / sinHalfAngle

        return Vec3(self.x * s, self.y * s, self.z * s)

    # spherical linear interpolation between rotations
    def slerp(self, b: "Quat", alpha: float):
        delta = (b * self.inverse()).to_rotation_vector()
        quat_tmp = Quat.from_rotation_vector(delta * alpha)
        return (quat_tmp * self).normalized()

    """
    constructors + serialization
    """

    @staticmethod
    def identity():
        return Quat(0, 0, 0, 1)

    @staticmethod
    def from_rotation_vector(v: Vec3):
        angleSquared = v.length_squared()
        s = 0
        c = 1
        if angleSquared > 0:
            angle = math.sqrt(angleSquared)
            s = math.sin(angle * 0.5) / angle
            c = math.cos(angle * 0.5)

        return Quat(s * v.x, s * v.y, s * v.z, c)

    @staticmethod
    # rotates ccw around a specified axis
    def from_axis_angle(axis: Vec3, angle_rads: float):
        # don't divide by zero
        if axis.length_squared == 0:
            if angle_rads == 0:
                assert False
            else:
                return Quat(0, 0, 0, 1)

        unitAxis = axis.normalized()
        sinHalfAngle = math.sin(angle_rads * 0.5)

        return Quat(
            unitAxis.x * sinHalfAngle,
            unitAxis.y * sinHalfAngle,
            unitAxis.z * sinHalfAngle,
            math.cos(angle_rads * 0.5),
        )

    @staticmethod
    # axis: x = 0, y = 1, z = 2
    def from_rotation_on_axis(axis: int, angle_rads: float):
        if axis == 0:
            return Quat.from_axis_angle(Vec3(1, 0, 0), angle_rads)
        elif axis == 1:
            return Quat.from_axis_angle(Vec3(0, 1, 0), angle_rads)
        elif axis == 2:
            return Quat.from_axis_angle(Vec3(0, 0, 1), angle_rads)
        else:
            assert False  # axis doesn't exist

    @staticmethod
    def from_mat3_numpy(mat3: np.ndarray):
        mat3 = mat3.squeeze()
        trace = mat3[0][0] + mat3[1][1] + mat3[2][2]

        # trace should almost always be positive
        if trace > 0:
            s = math.sqrt(trace + 1) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (mat3[2][1] - mat3[1][2]) / s
            y = (mat3[0][2] - mat3[2][0]) / s
            z = (mat3[1][0] - mat3[0][1]) / s
        elif (mat3[0][0] > mat3[1][1]) and (mat3[0][0] > mat3[2][2]):
            s = math.sqrt(1 + mat3[0][0] - mat3[1][1] - mat3[2][2]) * 2
            w = (mat3[2][1] - mat3[1][2]) / s
            x = 0.25 * s
            y = (mat3[0][1] + mat3[1][0]) / s
            z = (mat3[2][0] + mat3[0][2]) / s
        elif mat3[1][1] > mat3[2][2]:
            s = math.sqrt(1 + mat3[1][1] - mat3[0][0] - mat3[2][2]) * 2  # s = 4 * qy
            w = (mat3[0][2] - mat3[2][0]) / s
            x = (mat3[0][1] + mat3[1][0]) / s
            y = 0.25 * s
            z = (mat3[1][2] + mat3[2][1]) / s
        else:
            s = math.sqrt(1 + mat3[2][2] - mat3[0][0] - mat3[1][1]) * 2  # s = 4 * qz
            w = (mat3[1][0] - mat3[0][1]) / s
            x = (mat3[0][2] + mat3[2][0]) / s
            y = (mat3[1][2] + mat3[2][1]) / s
            z = 0.25 * s

        return Quat(x, y, z, w)

    def to_mat3_numpy(self):
        assert self.is_normalized()  # quat math bug

        tx = self.x + self.x
        ty = self.y + self.y
        tz = self.z + self.z

        twx = self.w * tx
        twy = self.w * ty
        twz = self.w * tz

        txx = self.x * tx
        txy = self.x * ty
        txz = self.x * tz

        tyy = self.y * ty
        tyz = self.y * tz
        tzz = self.z * tz

        mat3 = np.zeros((3, 3))
        mat3[0][0] = 1 - (tyy + tzz)
        mat3[0][1] = txy - twz
        mat3[0][2] = txz + twy
        mat3[1][0] = txy + twz
        mat3[1][1] = 1 - (txx + tzz)
        mat3[1][2] = tyz - twx
        mat3[2][0] = txz - twy
        mat3[2][1] = tyz + twx
        mat3[2][2] = 1 - (txx + tyy)
        return mat3

    @staticmethod
    def from_xyzw_numpy(xyzw: np.ndarray):
        xyzw = xyzw.squeeze()
        return Quat(xyzw[0], xyzw[1], xyzw[2], xyzw[3])

    def to_xyzw_numpy(self):
        return np.array([self.x, self.y, self.z, self.w])

    @staticmethod
    def from_numpy(np_arr: np.ndarray):
        # we try and be smart and detect what representation
        # you're passing in based on shape
        np_arr = np_arr.squeeze()

        if np_arr.shape == (3, 3):
            return Quat.from_mat3_numpy(np_arr)
        elif np_arr.shape == (4,):
            return Quat.from_xyzw_numpy(np_arr)
        else:
            assert False  # what kind of numpy are you passing in
