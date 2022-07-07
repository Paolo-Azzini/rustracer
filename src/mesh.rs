//! Triangle Mesh module
//!
//! Provides tools to define and manage triangle meshes.
//! Like [shapes](module@shape), [`Mesh`](struct@Mesh) implement the
//! [`RayIntersection`](trait@RayIntersection) trait.

use crate::misc::Vector2D;
use crate::normal::Normal;
use crate::point::Point;
use crate::vector::Vector;
use obj::Vertex;

/// Geometrical shape corresponding to a triangle.
/// This shape isn't meant to be used by itself, but only as a building block for [`Mesh`](struct@Mesh).
#[derive(Debug)]
pub struct Triangle {
    /// A triangle is defined by its three vertices.
    vertices: (Vertex, Vertex, Vertex),
}

impl Triangle {
    /// Provides a constructor for [`Triangle`](struct@Triangle).
    pub fn new(v1: Vertex, v2: Vertex, v3: Vertex) -> Self {
        Triangle {
            vertices: (v1, v2, v3),
        }
    }

    /// Get (u,v) coordinates of a 3D point lying on the triangle.
    pub fn get_point_to_uv(&self, p: Point) -> Vector2D {
        let a = Point::from(self.vertices.0.position);
        let b = Point::from(self.vertices.1.position);
        let c = Point::from(self.vertices.2.position);
        let m_ap = (p.y - a.y) / (p.x - a.x);
        let m_bc = (c.y - b.y) / (c.x - b.x);
        let q_ap = a.y - m_ap * a.x;
        let q_bc = b.y - m_bc * b.x;
        let x_q = (q_bc - q_ap) / (m_ap - m_bc);
        Vector2D {
            u: (p.x - a.x) / (x_q - a.x),
            v: (x_q - b.x) / (c.x - b.x),
        }
    }

    /// Calculates normals to [`Triangle`](struct@Triangle)'s surface.
    ///
    /// This Function is meant to be used inside [`Triangle`](struct@Triangle)'s
    /// [`RayIntersection`](trait@RayIntersection) implementation.\
    /// `ray_dir` is the direction of an impacting [`Ray`](struct@Ray) and\
    /// is used to determine on which side of the surface the normal is calculated.
    pub fn get_normal_flat(&self, ray_dir: Vector) -> Normal {
        let p1 = Point::from(self.vertices.0.position);
        let p2 = Point::from(self.vertices.1.position);
        let p3 = Point::from(self.vertices.2.position);
        let result = (p2 - p1) * (p3 - p1);
        if result.dot(ray_dir) < 0.0 {
            Normal::from(result)
        } else {
            Normal::from(result.neg())
        }
    }

    /// Calculates interpolated normals to [`Triangle`](struct@Triangle)'s surface.
    ///
    /// The normal is evaluated as interpolation of vertex's normals,
    /// to make the triangle surface look "rounded".
    ///
    /// This Function is meant to be used inside [`Triangle`](struct@Triangle)'s
    /// [`RayIntersection`](trait@RayIntersection) implementation.\
    /// `ray_dir` is the direction of an impacting [`Ray`](struct@Ray) and\
    /// is used to determine on which side of the surface the normal is calculated.
    pub fn get_normal_rounded(&self, uv: Vector2D, ray_dir: Vector) -> Normal {
        let n1 = Normal::from(self.vertices.0.normal);
        let n2 = Normal::from(self.vertices.1.normal);
        let n3 = Normal::from(self.vertices.2.normal);
        let result = n1 * (1. - uv.u) + n2 * uv.u * uv.v + n3 * uv.u * (1. - uv.v);
        if Vector::from(result).dot(ray_dir) < 0.0 {
            result
        } else {
            result.neg()
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::misc::IsClose;

    #[test]
    fn test_triangle() {
        let tri = Triangle::new(
            Vertex {
                position: [0., 0., 1.],
                normal: [0., 0., 1.],
            },
            Vertex {
                position: [1., 0., 0.],
                normal: [1., 0., 0.],
            },
            Vertex {
                position: [0., 1., 0.],
                normal: [0., 1., 0.],
            },
        );
        let p = Point::from((0.25, 0.25, 0.5));
        let uv = tri.get_point_to_uv(p);
        let ray_dir = Vector::from((0., 0., -1.));
        assert!(tri
            .get_normal_flat(ray_dir)
            .is_close(Normal::from((1., 1., 1.))));
        assert!(tri
            .get_normal_rounded(uv, ray_dir)
            .is_close(Normal::from((0.25, 0.25, 0.5))))
    }
}
