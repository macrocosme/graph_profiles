from pandas import DataFrame
import numpy as np
import copy

class Graph:
    """Class repressenting a graph.

    Usage
    -----
    >>> g = Graph()
    >>> g.add_edge(0, 1, 10)
    >>> g.add_edge(0, 2, 6)
    >>> g.add_edge(0, 3, 5)
    >>> g.add_edge(1, 3, 15)
    >>> g.add_edge(2, 3, 4)
    >>> g.spanning_tree()
    >>> g.get_result()
    >>> g.get_nodes_degrees()
    >>> g.as_dataframe()
    """

    def __init__(self):
        self.V = 0 # Number of vertices
        self.V_no_nan = 0 # V for mst after removing w=np.inf vases
        self.vertices = []
        self.graph = [] # list storing our graph
        self.mst = [] # Minumum spanning tree
        self.longest_path = None
        self.centrality = None
        self.farness = None

    def add_edge(self, u, v, w):
        """Add edge to graph.
        """
        self.graph.append([u,v,w])

        self.add_vertex(u)
        self.add_vertex(v)

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)
        self.V = len(self.vertices)

    def find(self, parent, i):
        """Find set of an element i via path compression technique.
        """
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        """Union of two sets of x and y via union by rank.
        """
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1

    def spanning_tree(self, reverse=False):
        """Construct a (minimum) spanning tree using Kruskal's algorithm.

        Kruskal's minimum spanning tree algorithm has a time complexity of O(ElogE) or O(ElogV).

        when reverse == False, returns the mimimum spanning tree
        when reverse == True, returns the maximum spanning tree

        Steps
        -----
        1) Sort edges. 2) Find-union.

        Big O analysis
        --------------
        Sorting edges has a time complexity of O(ELogE). Once the edges are sorted, the algorithm iterates
        through all edges and apply the find-union algorithm. The find and union operations have a time complexity
        of O(LogV). Hence, the overall complexity is O(ELogE + ELogV) time. The value of E can be at most O(V^2),
        so O(LogV) and O(LogE) are equivalent. Therefore, overall time complexity is O(ElogE) or O(ElogV).
        """
        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for self.mst[]

        # Step 1:   Sort all the edges in increasing order of their weight.
        #           If we are not allowed to change the given graph,
        #           we can create a copy of graph
        self.graph = sorted(self.graph, key=lambda item: item[2], reverse=reverse)
        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1 :

            # Step 2: Pick the smallest edge and increment
                    # the index for next iteration
            u, v, w =  self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent ,v)

            # If including this edge does't cause cycle,
            # - include it in self.mst
            # - increment the index of self.mst for next edge
            if x != y:
                e = e + 1
                self.mst.append([u,v,w])
                self.union(parent, rank, x, y)


        # remove np.inf distances from tree
        _mst = copy.deepcopy(self.mst)
        while _mst[-1][-1] == np.inf:
            _mst.pop()
        self.V_no_nan = np.unique(
            np.append(
                np.unique(np.array(_mst).T[0]),
                np.unique(np.array(_mst).T[1])
            )
        ).size

        print (self.V, self.V_no_nan)

    def get_mst_edges_dict(self):
        mst_vertices = {}
        for s in self.mst:
            try:
                mst_vertices[s[0]].append(s[1])
            except KeyError:
                mst_vertices[s[0]] = []
                mst_vertices[s[0]].append(s[1])
            try:
                mst_vertices[s[1]].append(s[0])
            except KeyError:
                mst_vertices[s[1]] = []
                mst_vertices[s[1]].append(s[0])

        return mst_vertices

    def breadth_first_search(self, s):
        mst_vertices = self.get_mst_edges_dict()
        path = []
        distances = [-1] * (len(mst_vertices))

        queue = []
        queue.append(s)
        path.append(s)
        distances[s] = 0

        while queue:
            s = queue.pop(0)
            for i in mst_vertices[s]:
                if distances[i] == -1:
                    queue.append(i)
                    distances[i] = distances[s] + 1

        max_dist = 0
        node_idx = 0
        path = {}

        for i, dist in enumerate(distances):
            path[i] = dist
            if dist > max_dist:
                max_dist = dist
                node_idx = i

        path = sorted(path.items(), key=lambda kv: kv[1])

        return [node_idx, max_dist, path]

    def construct_sequence(self, s, t, path=[]):
        path.append(s)
        for n in self.get_mst_edges_dict()[s]:
            if n == t:
                path.append(n)
                return True, path
            else:
                if n not in path:
                    found, path = self.construct_sequence(n, t, path)
                    if not found:
                        path.pop()
                    else:
                        return True, path

        return False, path

    def get_longest_path(self):
        if self.longest_path is None:
            self.compute_longest_path()
        return self.longest_path

    def compute_longest_path(self):
        node_id, d, path = self.breadth_first_search(self.find_least_connected_vertex())
        node_id, d, nodes_distance = self.breadth_first_search(node_id)
        nodes_distance = np.array(nodes_distance)
        start = nodes_distance[
                np.where(nodes_distance.T[1] == nodes_distance.T[1].min())
            ].T[0][0]
        target = nodes_distance[
                np.where(nodes_distance.T[1] == nodes_distance.T[1].max())
            ].T[0][0]

        self.longest_path = self.construct_sequence(start,target, [])[1]

    def find_least_connected_vertex(self):
        return np.where(
            np.equal(
                self.get_closeness_centrality(),
                np.nanmin(self.get_closeness_centrality())
            )
        )[0][0]

    def get_closeness_centrality(self):
        if self.centrality is None:
            return self.compute_closeness_centrality()

        return self.centrality

    def compute_closeness_centrality(self):
        mst = np.array(self.mst).T
        N = self.V
        k = self.V_no_nan-1
        self.farness = np.empty([N, N])
        self.farness[:] = np.NaN

        for s in range(N):
            _, _, path = self.breadth_first_search(s)
            for e, d in path:
                w = mst[2][np.where(
                    ((mst[0] == s) & (mst[1] == e)) |
                    ((mst[1] == s) & (mst[0] == e))
                )]
                if len(w) > 0 and w != np.inf:
                    self.farness[s, e] = d

        self.centrality = k/np.nansum(self.farness, axis=1)
        return self.centrality

    def elongation(self):
        """Compute the graph elongation

        Returns:
            elongation : float
                e = length / estimated half width
            normalized_elongation : float
                e/N, with N the number of vertices in the graph.
        """
        if self.farness is None:
            _ = self.compute_closeness_centrality()

        l = self.length()
        w = self.estimate_half_width()

        return l/w, l/(w*self.V_no_nan)

    def length(self):
        if self.farness is None:
            _ = self.compute_closeness_centrality()

        return np.nanmean(self.farness, axis=1)[self.find_least_connected_vertex()]

    def estimate_half_width(self, start_node=0):
        """Estimate of graph half width
        """
        return np.nanmean(
            np.unique(
                self.farness[self.find_least_connected_vertex()],
                return_counts=True
            )[1]
        )

        # def traverse(longest_path, mst_vertices, vertex, level, width_level, visited):
        #     if level not in width_level.keys():
        #         width_level[level] = []
        #     for neighbour in mst_vertices[vertex]:
        #         if neighbour not in longest_path and neighbour not in visited:
        #             width_level[level].append(len(mst_vertices[neighbour]))
        #             visited.append(neighbour)
        #             width_level, visited = traverse(longest_path,
        #                                    mst_vertices,
        #                                    neighbour,
        #                                    level+1,
        #                                    width_level,
        #                                    visited)
        #     return width_level, visited
        #
        # mst_vertices = self.get_mst_edges_dict()
        # longest_path = self.get_longest_path()
        #
        # width_level = {}
        # for level, vertex in enumerate(longest_path):
        #     if level not in width_level.keys():
        #         width_level[level] = []
        #     width_level[level].append(len(mst_vertices[vertex]))
        #     visited = [vertex]
        #     width_level, visited = traverse(longest_path,
        #                            mst_vertices,
        #                            vertex,
        #                            level+1,
        #                            width_level,
        #                            visited)
        #
        # means = []
        # for level in width_level.keys():
        #     if len(width_level[level]) != 0:
        #         means.append(np.mean(width_level[level]))
        # return np.mean(means), width_level

    def graph_as_dataframe(self):
        df = DataFrame(self.graph, columns=['u', 'v', 'w'])
        df = df.astype(dtype={'u': int, 'v': int, 'w': float})
        return df

    def as_dataframe(self):
        df = DataFrame(self.mst, columns=['u', 'v', 'w'])
        df = df.astype(dtype={'u': int, 'v': int, 'w': float})
        return df

    def get_nodes_degrees(self):
        df = self.as_dataframe()
        degrees = {}
        degrees_u = df.groupby('u').size()
        degrees_v = df.groupby('v').size()
        for key in degrees_u.keys():
            if key not in degrees.keys():
                degrees[key] = degrees_u[key]
            else:
                degrees[key] += degrees_u[key]

        for key in degrees_v.keys():
            if key not in degrees.keys():
                degrees[key] = degrees_v[key]
            else:
                degrees[key] += degrees_v[key]

        df_degrees = DataFrame([[k, degrees[k]] for k in degrees.keys()], columns=['vertex', 'degree'])
        return df_degrees.astype(dtype={'vertex': int, 'degree': int})

    def construct_path(im, df_mst, verbose=False):
        if verbose:
            print('Construct sequence')
        used = []
        path = []

        # Start building sequence
        index_u = int(df_mst.loc[df_mst['w'] == np.min(df_mst['w']), 'u'])
        index_v = int(df_mst.loc[df_mst['w'] == np.min(df_mst['w']), 'v'])
        path.append(im[index_u].copy())
        path.append(im[index_v].copy())
        used.append(index_u)
        _continue = True
        while _continue:
            _found = False

            # find next best link for index_v
            for candidate in df_mst.loc[(df_mst['v'] == index_v) & (df_mst['u'] != index_u)].sort_values(by=['w']):
                if not _found and int(candidate['v']) not in used:
                    index_v = index_v
                    index_u = int(candidate['u'])
                    path.append(im[index_u].copy())
                    _found = True

            # exit clause
            if _found == False:
                _continue = False

        return path

    def which_has_least_degree(self, u, v):
        degrees = self.get_nodes_degrees()
        deg_u = degrees.loc[degrees['vertex'] == u, 'degree'].values[0]
        deg_v = degrees.loc[degrees['vertex'] == v, 'degree'].values[0]
        return u if deg_u < deg_v else v if deg_u > deg_v else u

    def define_path_starting_vertices(self):
        degrees = self.get_nodes_degrees()
        df = self.as_dataframe()

        u, v, w = df.loc[
            df['u'].isin(
                np.asarray(
                    degrees.loc[degrees['degree'] == np.min(degrees['degree']), 'vertex'].values
                )
            )
        ].values[0]

        root = self.which_has_least_degree(u, v)
        leaf = u if u != root else v

        return int(root), int(leaf)

    def get_result(self):
        """Print results to display the built MST
        """
        print("Minimum spanning tree's edges:")
        for u, v, w in self.mst:
            print ("%d -- %d == %.2f" % (u, v, w))

def make_test_graph():
    g = Graph()
    g.add_edge(0, 1, 10)
    g.add_edge(0, 2, 6)
    g.add_edge(0, 3, 5)
    g.add_edge(1, 3, 15)
    g.add_edge(2, 3, 4)
    g.add_edge(2, 4, 3)
    g.add_edge(1, 4, 2)
    g.add_edge(0, 4, 2)
    g.add_edge(3, 4, 2)
    g.add_edge(2, 3, 4)
    g.add_edge(2, 5, 3)
    g.add_edge(1, 5, 2)
    g.add_edge(0, 5, 2)
    g.spanning_tree()
    return g
