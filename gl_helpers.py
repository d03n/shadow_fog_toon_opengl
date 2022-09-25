from xml.etree.ElementTree import PI
from numpy import array, ndarray, zeros, dot, cross, float32, identity
from numpy.linalg import norm
from math import sqrt, sin, cos, tan, acos, pi

def Identity():
    return array(((1, 0, 0, 0),
                  (0, 1, 0, 0),
                  (0, 0, 1, 0),
                  (0, 0, 0, 1)), dtype=float32)

def normalize(v):
    l = norm(v)
    if l == 0:
        return v
    else:
        return v/l

def Translate(tx, ty, tz):
    return array(((1, 0, 0, tx),
                  (0, 1, 0, ty),
                  (0, 0, 1, tz),
                  (0, 0, 0, 1)), dtype=float32)

def Scale(sx, sy, sz):
    return array(((sx, 0, 0, 0),
                  (0, sy, 0, 0),
                  (0, 0, sz, 0),
                  (0, 0, 0, 1)), dtype=float32)

# def Rotate(angle, x, y, z):
#     angle = pi*angle/180
#     sqr_a = x*x
#     sqr_b = y*y
#     sqr_c = z*z
#     len2 = sqr_a + sqr_b + sqr_c
#     k2 = cos(angle)
#     k1 = (1.0-k2) / len2
#     k3 = sin(angle) / sqrt(len2)
#     k1ab = k1*x*y
#     k1ac = k1*x*z
#     k1bc = k1*y*z
#     k3a = k3*x
#     k3b = k3*y
#     k3c= k3*z
#     return array(((k1*sqr_a+k2, k1ab-k3c,k1ac+k3b, 0),
#                   (k1ab+k3c, k1*sqr_b+k2, k1bc-k3a, 0),
#                   (k1ac-k3b, k1bc+k3a, k1*sqr_c+k2, 0),
#                   (0, 0, 0, 1)), dtype=float32)

def Rotate(theta, x, y, z):
    theta = pi*theta/180
    len = sqrt(x*x + y*y + z*z)
    x /= len
    y /= len
    z /= len
    xx = x*x
    yy = y*y
    zz = z*z

    C = cos(theta)
    S = sin(theta)

    return array(((x*x*(1-C)+C, x*y*(1-C)-z*S, x*z*(1-C)+y*S, 0),
                     (y*x*(1-C)+z*S, y*y*(1-C)+C, y*z*(1-C)-x*S, 0),
                     (z*x*(1-C)-y*S, z*y*(1-C)+x*S, z*z*(1-C)+C, 0),
                     (0, 0, 0, 1)), dtype=float32)

def LookAt(eyex, eyey, eyez, atx, aty, atz, upx, upy, upz):
    eye = array((eyex, eyey, eyez))
    at = array((atx, aty, atz))
    up = array((upx, upy, upz))

    Z = normalize(eye - at)
    Y = normalize(up)
    X = normalize(cross(Y,Z))
    Y = normalize(cross(Z,X))

    return array(((X[0], X[1], X[2], -dot(X,eye)),
                  (Y[0], Y[1], Y[2], -dot(Y,eye)),
                  (Z[0], Z[1], Z[2], -dot(Z,eye)),
                  (0, 0, 0, 1)), dtype=float32)

def Perspective(fovy, aspect, zNear, zFar):
    fovy = pi * fovy / 180
    return array(((1/tan(fovy/2)/aspect, 0, 0, 0),
                  (0, 1 / tan(fovy/2), 0, 0),
                  (0, 0, -(zFar+zNear)/(zFar-zNear), -(2*zNear*zFar)/(zFar-zNear)),
                  (0, 0, -1, 0)), dtype=float32)

def Frustum(left, right, bottom, top, near, far):
    return array(((2*near/(right-left), 0, (right+left)/(right-left), 0),
                  (0, 2*near/(top-bottom), (top+bottom)/(top-bottom), 0),
                  (0, 0, -(far+near)/(far-near), -2*far*near/(far-near)),
                  (0, 0, -1, 0)), dtype=float32)

def Ortho(left, right, bottom, top, near, far):
    return array(((2 / (right-left), 0, 0, -((right+left) / (right-left))),
                  (0, 2 / (top - bottom), 0, -((top+bottom) / (top-bottom))),
                  (0, 0, -2/(far-near), -((far+near) / (far-near))),
                  (0, 0, 0, 1)), dtype=float32)