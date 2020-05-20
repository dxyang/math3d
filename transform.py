import numpy as np

from quat import Quat
from vec3 import Vec3


class Transform:
    """
    Position and orientation combined.
    """

    def __init__(
        self, orientation: Quat = Quat.identity(), position: Vec3 = Vec3.zero()
    ):
        self.rotation = orientation
        self.translation = position

    """
    python overrides
    """

    def __str__(self):
        printstr = "Rot: {} \n Translation: {}".format(
            self.rotation.to_mat3_numpy(), self.translation.to_numpy()
        )
        return printstr

    def __repr__(self):
        return 'Transform \n {}'.format(self.__str__())

    def __mul__(self, rhs: "Transform"):
        t = type(rhs)
        if t is Transform:
            new_rotation = self.rotation * rhs.rotation
            new_translation = self.transform(rhs.translation)
            return Transform(new_rotation, new_translation)
        elif t is Vec3:
            return self.transform(rhs)
        else:
            assert False  # don't * not transforms with this

    def __eq__(self, rhs: "Transform"):
        return (self.translation == rhs.translation) and (self.rotation == rhs.rotation)

    def __ne__(self, rhs: "Transform"):
        return (self.translation != rhs.translation) or (self.rotation != rhs.rotation)

    """
    useful functions
    """

    def rotate(self, v: Vec3):
        return self.rotation.rotate(v)

    def inverse_rotate(self, v: Vec3):
        return self.rotation.inverse_rotate(v)

    def translate(self, v: Vec3):
        return v + self.translation

    def transform(self, v: Vec3):
        return self.rotate(v) + self.translation

    def inverse_transform(self, v: Vec3):
        return self.inverse_rotate(v - self.translation)

    def inverse(self):
        inv = self.rotation.inverse()
        new_translation = inv.rotate(-self.translation)
        return Transform(inv, new_translation)

    def normalize(self):
        self.rotation.normalize()

    def normalized(self):
        return Transform(self.rotation.normalized(), self.translation)

    def is_normalized(self):
        return self.rotation.is_normalized()

    def is_nan(self):
        return self.rotation.is_nan() or self.translation.is_nan()

    def is_finite(self):
        return self.rotation.is_finite() and self.translation.is_finite()

    """
    constructors + serialization
    """

    @staticmethod
    def identity():
        return Transform(Quat.identity(), Vec3.zero())

    def to_quat_vec3_numpy(self):
        return np.array(
            [
                self.rotation.x,
                self.rotation.y,
                self.rotation.z,
                self.rotation.w,
                self.translation.x,
                self.translation.y,
                self.translation.z,
            ]
        )

    @staticmethod
    def from_quat_vec3_numpy(quatvec3: np.ndarray):
        return Transform(Quat.from_numpy(quatvec3[:4]), Vec3.from_numpy(quatvec3[4:]))

    def to_mat4_numpy(self):
        mat3 = self.rotation.to_mat3_numpy()
        vec3 = self.translation.to_numpy()

        mat1 = np.c_[mat3, vec3]
        mat2 = np.array([0, 0, 0, 1]).reshape((1, 4))
        trans_matrix = np.vstack([mat1, mat2])

        return trans_matrix

    @staticmethod
    def from_mat4_numpy(mat4: np.ndarray):
        return Transform(Quat.from_numpy(mat4[:3, :3]), Vec3.from_numpy(mat4[:3, 3]))

    @staticmethod
    def from_numpy(np_arr: np.ndarray):
        # we try to do something smart based on the shape
        np_arr = np_arr.squeeze()

        if np_arr.shape == (7,):
            return Transform.from_quat_vec3_numpy(np_arr)
        elif np_arr.shape == (4, 4):
            return Transform.from_mat4_numpy(np_arr)
