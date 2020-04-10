import math
import numpy as np

from _constants import MATH_EPS


class Vec3:
    """
    Vec3 represents a 3-dimensional vector or point
    in space, consisting of coordinates x, y and z.
    """

    def __init__(self, _x: float = 0, _y: float = 0, _z: float = 0):
        self.x = _x
        self.y = _y
        self.z = _z

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
        else:
            assert False  # vec3 only has 3 things

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        else:
            assert False  # vec3 only has 3 things

    def __str__(self):
        return "x: {}, y: {}, z:{}".format(self.x, self.y, self.z)

    def __repr__(self):
        return 'Vec3({}, {}, {})'.format(self.x, self.y, self.z)

    def __add__(self, rhs: "Vec3"):
        return Vec3(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

    def __iadd__(self, rhs: "Vec3"):
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z
        return self

    def __sub__(self, rhs: "Vec3"):
        return Vec3(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)

    def __isub__(self, rhs: "Vec3"):
        self.x -= rhs.x
        self.y -= rhs.y
        self.z -= rhs.z
        return self

    def __mul__(self, scalar: float):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __imul__(self, scalar: float):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        return self

    def __truediv__(self, scalar: float):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __idiv__(self, scalar: float):
        self.x /= scalar
        self.y /= scalar
        self.z /= scalar
        return self

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, rhs: "Vec3"):
        return (
            (abs(self.x - rhs.x) <= MATH_EPS)
            and (abs(self.y - rhs.y) <= MATH_EPS)
            and (abs(self.z - rhs.z) <= MATH_EPS)
        )

    def __ne__(self, rhs: "Vec3"):
        return (
            (abs(self.x - rhs.x) > MATH_EPS)
            or (abs(self.y - rhs.y) > MATH_EPS)
            or (abs(self.z - rhs.z) > MATH_EPS)
        )

    """
    useful functions
    """

    def dot(self, rhs: "Vec3"):
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    def cross(self, rhs: "Vec3"):
        return Vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )

    def length_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.length_squared())

    def distance_squared(self, b: "Vec3"):
        delta = b - self
        return delta.length_squared()

    def distance(self, b: "Vec3"):
        delta = b - self
        return delta.length()

    def is_normalized(self):
        return abs(self.length() - 1.0) <= MATH_EPS

    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length

    def normalized(self):
        return self / self.length()

    def project_to(self, b: "Vec3"):
        l2 = b.length_squared()
        return b * (self.dot(b) / l2)

    def project_to_plane(self, normal: "Vec3"):
        return self - self.project_to(normal)

    def is_nan(self):
        return not np.isfinite(self.x + self.y + self.z)

    def is_finite(self):
        return np.isfinite(self.x + self.y + self.z)

    """
    constructors + serialization
    """

    @staticmethod
    def from_numpy(np_arr: np.ndarray):
        np_arr = np_arr.squeeze()
        assert len(np_arr) == 3
        return Vec3(np_arr[0], np_arr[1], np_arr[2])

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def zero():
        return Vec3(0, 0, 0)

    @staticmethod
    def x_axis():
        return Vec3(1, 0, 0)

    @staticmethod
    def y_axis():
        return Vec3(0, 1, 0)

    @staticmethod
    def z_axis():
        return Vec3(0, 0, 1)
