import stl
import numpy as np
from Geometry import SurfaceGeometry3D


def generate_sphere(file_path, radius=1, center=(0, 0, 0), num_points=20):
    import pyvista as pv
    sphere = pv.Sphere(radius, center=center, theta_resolution=num_points, phi_resolution=num_points)
    mesh_obj = sphere.triangulate()
    mesh_obj.save(file_path)


class ComposeGeometry(object):
    def __init__(self,  stl_file):

        self.vectors = stl.mesh.Mesh.from_file(stl_file).vectors

        self.geo = SurfaceGeometry3D(self.vectors)

        if not self.geo.boundary_edges:
            raise Exception("Your geometry is a closed volume. This is invalid geometry to proceed!")

        if len(self.geo.boundary_nodes) == 1:
            self.boundary_nodes = self.geo.boundary_nodes[0][::-1]
        else:
            raise Exception("The geometry is invalid. There are more than one perimeter existed!")

        self.output_file_name = stl_file[0:-4] + "_skirted.stl"

    def __find_master_plane(self, offset):

        boundary_vertices = self.geo.vertices[self.boundary_nodes]

        origin = self.geo.centroid(boundary_vertices)
        _, _, v = np.linalg.svd(boundary_vertices - origin)
        normal = v[2]

        # Coefficients should point out of the protruding part
        normal = -normal  # This requires automation

        basis = self.geo.find_normal_vectors_of_a_surface(normal)

        boundary_nodes_transformed = self.geo.transform_points(boundary_vertices,
                                                               self.geo.global_origin, self.geo.global_basis,
                                                               origin, basis)
        origin_temp = boundary_vertices[np.argmax(boundary_nodes_transformed[:, 2])]

        origin = self.geo.project_to_plane(origin, origin_temp, normal) + offset * normal

        return origin, basis

    def add_skirt(self, offset):

        mst_origin, mst_basis = self.__find_master_plane(offset)

        boundary_vertices = self.geo.vertices[self.boundary_nodes]
        projected_vertices = self.geo.project_to_plane(boundary_vertices, mst_origin, mst_basis[2])

        new_vertices = self.geo.transform_points(self.geo.vertices,
                                                 self.geo.global_origin, self.geo.global_basis,
                                                 mst_origin, mst_basis)
        new_projected_vertices = self.geo.transform_points(projected_vertices,
                                                           self.geo.global_origin, self.geo.global_basis,
                                                           mst_origin, mst_basis)

        new_boundary_nodes = np.array([i + self.geo.n_vertices for i in range(len(new_projected_vertices))])

        projected_faces = [new_boundary_nodes[0:-1], new_boundary_nodes[1:], self.boundary_nodes[1:]]
        projected_faces = np.array(projected_faces).T
        new_faces = np.vstack([self.geo.faces, projected_faces])
        projected_faces = [self.boundary_nodes[-1:0:-1], self.boundary_nodes[-2::-1], new_boundary_nodes[-2::-1]]
        projected_faces = np.array(projected_faces).T
        new_faces = np.vstack([new_faces, projected_faces])

        projected_faces = [[new_boundary_nodes[-1], new_boundary_nodes[0], self.boundary_nodes[0]],
                           [self.boundary_nodes[0], self.boundary_nodes[-1], new_boundary_nodes[-1]]]
        projected_faces = np.array(projected_faces)
        new_faces = np.vstack([new_faces, projected_faces])

        new_vertices = np.vstack([new_vertices, new_projected_vertices])

        new_vertices[:, 2] -= np.mean(new_projected_vertices[:, 2])

        new_mesh = stl.mesh.Mesh(np.zeros(new_faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        new_mesh.vectors = new_vertices[new_faces]
        new_mesh.save(self.output_file_name)

        print(self.output_file_name + " successfully created!")


if __name__ == '__main__':
    compGeo = ComposeGeometry('../Geometries/Part.stl')
    compGeo.add_skirt(0.0)
