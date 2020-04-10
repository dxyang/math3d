# math3d
A simple to use Python math library for SE3 operations useful in computer vision, robotics, and AR/VR

This library draws heavy inspiration from OVR_Math.h within the [Oculus PC SDK](https://developer.oculus.com).

This library includes the basic building blocks of a vector3 (`Vec3`), a quaternion (`Quat`), and a rigid body transform (`Transform`) consisting of a vector3 and quaternion. In my experience, such a representation is expressive enough for all needs and relatively efficient.

These classes all have convenience functions to other representations. Specifically, rotations can also be represented by 3x3 matrices, euler angles, and axis + angle. Similarly, a rigid body transform can be represented as a 4x4 matrix.

For transforms, overloading the python multiplication operators allows for trivial chaining of operations:
```python
T_A_B = Transform() # transform of B relative to A
T_B_C = Transform() # transform of C relative to B
T_A_C = T_A_B * T_B_C

C_p = Vec3() # point wrt to C
A_p = T_A_C * C_p
```

This is a continual pet project in development.

Recommended reading:
* [Representing Robot Pose: The good, the bad, and the ugly. - Paul Furgale](http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly)