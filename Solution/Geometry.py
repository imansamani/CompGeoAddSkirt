import numpy as np


class SurfaceGeometry3D(object):
    def __init__(self, vectors):

        self.global_origin = np.array([0, 0, 0])
        self.global_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        vid = 0
        hash_map = {}
        self.faces = []
        self.vertices = []
        for face in vectors:
            tri = []
            for v in face:
                key = tuple(v)
                if key in hash_map:
                    tri.append(hash_map[key])
                else:
                    self.vertices.append(v)
                    tri.append(vid)
                    hash_map[key] = vid
                    vid += 1
            self.faces.append(tri)
        self.faces = np.array(self.faces)
        self.n_faces = len(self.faces)
        self.vertices = np.array(self.vertices)
        self.n_vertices = len(self.vertices)

        self.edges = {}
        for n in range(self.n_faces):
            face = self.faces[n]
            for i in range(3):
                e = tuple(face[[i - 1, i]])
                if e not in self.edges:
                    self.edges[e[::-1]] = [n]
                else:
                    self.edges[e].append(n)

        self.boundary_edges = {}
        for e, faces in self.edges.items():
            if len(faces) == 1:
                v, w = sorted(e)
                if v in self.boundary_edges:
                    self.boundary_edges[v].append(w)
                else:
                    self.boundary_edges[v] = [w]
                if w in self.boundary_edges:
                    self.boundary_edges[w].append(v)
                else:
                    self.boundary_edges[w] = [v]
            elif len(faces) > 2:
                raise Exception("An intersecting plane detected in your geometry. Please fix that and come back!")

        self.boundary_nodes = self.find_cycles(self.boundary_edges)

    @staticmethod
    def centroid(points):
        return np.mean(points, axis=0)

    @staticmethod
    def find_cycles(graph):
        explored = {}
        cycles = []

        def dfs(v, parent):
            explored[v] = parent
            for w in graph[v]:
                if w not in explored:
                    dfs(w, v)
                elif w != parent and parent:
                    cycle = [w, v]
                    curr = explored[v]
                    while curr != w:
                        cycle.append(curr)
                        curr = explored[curr]
                    cycles.append(cycle)

        for v in graph:
            if v not in explored:
                dfs(v, None)

        return cycles

    @staticmethod
    def find_normal_vectors_of_a_surface(coefficients):
        a, b, c = coefficients
        if abs(a) < abs(b):
            vector = np.array([0, -c, b])
        else:
            vector = np.array([-c, 0, a])
        n3 = np.array([a, b, c])
        n2 = np.cross(vector, n3)
        n1 = np.cross(n2, n3)
        return np.array([n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2), n3 / np.linalg.norm(n3)])

    @staticmethod
    def find_scc(graph):
        print("This method still under development and will find the number of strongly connected components.")

    @staticmethod
    def project_to_plane(points, plane_point, plane_normal):
        d = np.dot((points - plane_point), plane_normal)
        projected_points = points - np.outer(d, plane_normal)
        return projected_points

    @staticmethod
    def transform_points(points, origin1, basis1, origin2, basis2):
        A = np.dot(basis1.T, basis2)
        T = origin2 - np.dot(A, origin1)
        return np.dot(A, points.T).T + T
