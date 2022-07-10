//! Triangle Mesh module
//!
//! Provides tools to define and manage triangle meshes.
//! Like [shapes](../shape), [`Mesh`](struct@Mesh) implements the
//! [`RayIntersection`](trait@RayIntersection) trait.

use crate::misc::Vector2D;
use crate::normal::Normal;
use crate::point::Point;
use crate::ray::Ray;
use crate::shape::{HitRecord, RayIntersection};
use crate::vector::Vector;
use crate::Pigment::Uniform;
use crate::BRDF::Specular;
use crate::{Color, Material, SpecularBRDF, Transformation, UniformPigment};
use obj::Vertex;

pub const E3: Vertex = Vertex {
    position: [0., 0., 1.],
    normal: [0., 0., 1.],
};
pub const E1: Vertex = Vertex {
    position: [1., 0., 0.],
    normal: [1., 0., 0.],
};
pub const E2: Vertex = Vertex {
    position: [0., 1., 0.],
    normal: [0., 1., 0.],
};
pub const NEG_E3: Vertex = Vertex {
    position: [0., 0., -1.],
    normal: [0., 0., -1.],
};
pub const NEG_E1: Vertex = Vertex {
    position: [-1., 0., 0.],
    normal: [-1., 0., 0.],
};
pub const NEG_E2: Vertex = Vertex {
    position: [0., -1., 0.],
    normal: [0., -1., 0.],
};

/// Triangles mesh.
/// This struct is meant to be build from a `.obj` file.
#[derive(Debug, Default)]
pub struct Mesh {
    /// list of triangles in the mesh
    pub triangles: Vec<Triangle>,
    /// transformation applied to the mesh
    pub transformation: Transformation,
    /// material of the mesh
    pub material: Material,
}

impl Mesh {
    /// Provides a constructor for [`Mesh`](struct@Mesh),
    /// return an empty `Mesh` with specified `Transformation` and `Material`.
    pub fn new(transformation: Transformation, material: Material) -> Self {
        Self {
            triangles: Vec::default(),
            transformation,
            material,
        }
    }

    /// Pushes a [`Triangle`](struct@Triangle) on the `Mesh`.
    pub fn push(&mut self, triangle: Triangle) {
        self.triangles.push(triangle);
    }
}

impl RayIntersection for Mesh {
    fn ray_intersection(&self, ray: Ray) -> Option<HitRecord> {
        let inv_ray = self.transformation.inverse() * ray;
        if inv_ray.dir.norm() < 1e-5 {
            return None;
        }
        let mut t = f32::INFINITY;
        let mut record: Option<HitRecord> = None;
        for tri in self.triangles.iter() {
            let hit = tri.get_ray_intersection(inv_ray);
            if hit.is_some() {
                let t1 = hit.unwrap().0;
                if t1 < t {
                    t = t1;
                    let uv_hit = hit.unwrap().1;
                    record = Some(HitRecord {
                        world_point: self.transformation * inv_ray.at(t),
                        surface_point: uv_hit,
                        normal: self.transformation * tri.get_normal_flat(inv_ray.dir),
                        t,
                        ray,
                        material: self.material.clone(),
                    });
                }
            }
        }
        record
    }
}

/// Geometrical shape corresponding to a triangle.
/// This shape isn't meant to be used by itself, but only as a building block for [`Mesh`](struct@Mesh).
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    /// A triangle is defined by its three vertices.
    vertices: (Vertex, Vertex, Vertex),
}

impl Triangle {
    /// Provides a constructor for [`Triangle`](struct@Triangle).
    pub fn new(a: Vertex, b: Vertex, c: Vertex) -> Self {
        Triangle {
            vertices: (a, b, c),
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
        let a = Point::from(self.vertices.0.position);
        let b = Point::from(self.vertices.1.position);
        let c = Point::from(self.vertices.2.position);
        let result = (b - a) * (c - a);
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
        let na = Normal::from(self.vertices.0.normal);
        let nb = Normal::from(self.vertices.1.normal);
        let nc = Normal::from(self.vertices.2.normal);
        let result = na * (1. - uv.u) + nb * uv.u * uv.v + nc * uv.u * (1. - uv.v);
        if Vector::from(result).dot(ray_dir) < 0.0 {
            result
        } else {
            result.neg()
        }
    }

    /// Evaluates the point of intersection with a [`Ray`](struct@Ray), if exists.
    ///
    /// This is a pseudo-implementation of the [`RayIntersection`](trait@RayIntersection) for triangles,
    /// it's meant to be used only inside the proper implementation of `RayIntersection` for [`Mesh`](struct@Mesh).
    fn get_ray_intersection(&self, ray: Ray) -> Option<(f32, Vector2D)> {
        let a = Point::from(self.vertices.0.position);
        let n = Vector::from(self.get_normal_flat(ray.dir));
        let det = -n.dot(ray.dir);
        let t = n.dot(ray.origin - a) / det;
        if (t <= ray.tmin) || (t >= ray.tmax) {
            return None;
        }
        let uv_hit = self.get_point_to_uv(ray.at(t));
        if uv_hit.u < 0. || uv_hit.u > 1. || uv_hit.v < 0. || uv_hit.v > 1. {
            return None;
        }
        Some((t, uv_hit))
    }
}

/// Returns an octahedron shaped mesh.
pub fn Ramiel(transformation: Transformation) -> Mesh {
    Mesh {
        triangles: vec![
            Triangle::new(E3, E1, E2),
            Triangle::new(E3, E2, NEG_E1),
            Triangle::new(E3, NEG_E1, NEG_E2),
            Triangle::new(E3, NEG_E2, E1),
            Triangle::new(NEG_E3, E2, E1),
            Triangle::new(NEG_E3, NEG_E1, E2),
            Triangle::new(NEG_E3, NEG_E2, NEG_E1),
            Triangle::new(NEG_E3, E1, NEG_E2),
        ],
        transformation,
        material: Material {
            brdf: Specular(SpecularBRDF {
                pigment: Uniform(UniformPigment {
                    color: Color::from((0., 0., 1.)),
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::misc::IsClose;

    #[test]
    fn test_triangle() {
        let tri = Triangle::new(E3, E1, E2);
        let p = Point::from((0.25, 0.25, 0.5));
        let uv = tri.get_point_to_uv(p);
        let ray_dir = Vector::from((0., 0., -1.));
        let ray = Ray {
            dir: ray_dir,
            origin: Point::from((0.25, 0.25, 1.)),
            ..Default::default()
        };
        let ray1 = Ray {
            dir: ray_dir,
            origin: Point::from((2., 2., 1.)),
            ..Default::default()
        };
        assert!(tri
            .get_normal_flat(ray_dir)
            .is_close(Normal::from((1., 1., 1.))));
        assert!(tri
            .get_normal_rounded(uv, ray_dir)
            .is_close(Normal::from((0.25, 0.25, 0.5))));
        assert!(
            matches!(tri.get_ray_intersection(ray), Some((t, uv_hit)) if t.is_close(0.5) && uv_hit.is_close(uv))
        );
        assert!(matches!(tri.get_ray_intersection(ray1), None))
    }

    #[test]
    fn test_mesh_ray_intersection() {
        let mesh = Ramiel(Transformation::default());
        let ray1 = Ray {
            dir: Vector::from((0., 0., -1.)),
            origin: Point::from((0.25, 0.25, 1.)),
            ..Default::default()
        };
        let ray2 = Ray {
            dir: Vector::from((0., 0., -1.)),
            origin: Point::from((-0.25, -0.25, 1.)),
            ..Default::default()
        };
        let ray3 = Ray {
            dir: Vector::from((0., 0., -1.)),
            origin: Point::from((2., 2., 1.)),
            ..Default::default()
        };
        let record1 = HitRecord {
            world_point: Point::from((0.25, 0.25, 0.5)),
            surface_point: Vector2D { u: 0.5, v: 0.5 },
            t: 0.5,
            ray: ray1,
            normal: Normal::from((1., 1., 1.)),
            material: Material::default(),
        };
        let record2 = HitRecord {
            world_point: Point::from((-0.25, -0.25, 0.5)),
            surface_point: Vector2D { u: 0.5, v: 0.5 },
            t: 0.5,
            ray: ray2,
            normal: Normal::from((-1., -1., 1.)),
            material: Material::default(),
        };
        assert!(matches!(mesh.ray_intersection(ray1), Some(record) if record.is_close(record1)));
        assert!(matches!(mesh.ray_intersection(ray2), Some(record) if record.is_close(record2)));
        assert!(matches!(mesh.ray_intersection(ray3), None));
    }
}
