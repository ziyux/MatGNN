import numpy as np
import torch

from dgl.convert import graph as dgl_graph
from dgl.transform import to_bidirected

from fe_utils.sph_fea import dist_emb, angle_emb, torsion_emb, xyz_to_dat


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConstructor(object):
    def __init__(self,
                 connect_method='CWC',
                 cutoff=5,
                 k=None,
                 strict=False,
                 gaussian_step=None,
                 num_spherical=None,
                 num_radial=None,
                 envelope_exponent=5):
        self.connect_method = connect_method
        self.cutoff = cutoff
        self.k = k
        self.strict = strict
        self.gaussian_step = gaussian_step
        if self.gaussian_step:
            self.ge = GaussianExpansion(0, cutoff, step=gaussian_step)
        self.use_sf = False
        if num_spherical and num_radial:
            self.sf = SphericalFeatures(num_spherical, num_radial, cutoff, envelope_exponent)
            self.use_sf = True

    def connect_graph(self, R):
        if self.connect_method == 'CWC':
            graph = self.connect_within_cutoff(R, self.cutoff, self.k, self.strict)
        elif self.connect_method == 'PBC':
            graph = self.periodic_boundary_condition(R, self.cutoff, self.k, self.strict)
        else:
            raise ValueError('Invalid graph connection method.')
        return graph

    def feature_assignment(self, graph, node_fea=None, edge_fea=None):
        if node_fea:
            for key, fea in node_fea.items():
                graph.ndata[key] = fea.to(device)
        if edge_fea:
            for key, fea in edge_fea.items():
                graph.edata[key] = fea.to(device)
        if self.use_sf:
            pos = graph.ndata['coords']
            edge_index = graph.edges()
            num_nodes = graph.num_nodes()
            dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)
            rbf, sbf, tbf = self.sf(dist, angle, torsion, idx_kj)
            idx = torch.concat((idx_ji.reshape(1, -1), idx_kj.reshape(1, -1)))
            sbf_sparse, tbf_sparse = map(lambda sf: torch.sparse_coo_tensor(idx, sf, (graph.num_edges(), graph.num_edges(), sf.size()[1])), (sbf, tbf))
            graph.edata['rbf'], graph.edata['sbf'], graph.edata['tbf'] = rbf, sbf_sparse, tbf_sparse
        return graph

    def periodic_boundary_condition(self, cell, cutoff, k=None, strict=None):
        cart_coords = []
        image = []
        index_dict = {}
        for i in range(len(cell)):
            cart_coords.append(cell[i].coords)
            image.append(i)
            index_dict[str(cell[i].species) + str(cell[i].coords)] = i

        if cutoff == np.inf:
            raise ValueError('Cutoff cannot be inf')
        all_nbr = cell.get_all_neighbors(cutoff)
        u, v = [], []
        for i in range(len(all_nbr)):
            nbrs = self.choose_k_nearest_PBC(all_nbr[i], k, strict) if k else all_nbr[i]
            v += [i] * len(nbrs)
            for nbr in nbrs:
                if str(nbr._species) + str(nbr.coords) in index_dict:
                    u.append(index_dict[str(nbr._species) + str(nbr.coords)])
                else:
                    cart_coords.append(nbr.coords)
                    image.append(nbr.index)
                    u.append(len(index_dict))
                    index_dict[str(nbr._species) + str(nbr.coords)] = u[-1]

        graph = dgl_graph((u, v))
        graph = to_bidirected(graph)
        graph = graph.to(device)

        graph.ndata['coords'] = torch.tensor(np.array(cart_coords), dtype=torch.float32, device=device)
        graph.ndata['image'] = torch.tensor(image, dtype=torch.int64, device=device)

        return graph

    def choose_k_nearest_PBC(self, nbrs, k, strict):
        if len(nbrs) > k:
            if strict:
                nbrs = sorted(nbrs, key=lambda x: x.nn_distance)[:k]
            else:
                dist = torch.tensor([nbr.nn_distance for nbr in nbrs], dtype=torch.float32)
                threshold = dist[dist.argsort()[k - 1]]
                nbrs = [nbrs[i] for i in torch.where(dist <= threshold)[0].tolist()]
        return nbrs

    def connect_within_cutoff(self, coords, cutoff, k=None, strict=None):
        coords = torch.tensor(coords, dtype=torch.float32)
        dist = torch.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        adj = (dist <= cutoff).long() - torch.eye(len(coords))
        if k is not None:
            adj = self.choose_k_nearest_CWC(adj, dist, k, strict)
        adj = adj + adj.T
        adj = adj.bool().to_sparse()
        u, v = adj.indices()
        graph = dgl_graph((u, v))
        graph = to_bidirected(graph)
        graph = graph.to(device)

        graph.ndata['coords'] = coords.to(device)
        dist = dist[adj.indices()[0], adj.indices()[1]].flatten().to(device)
        graph.edata['dist'] = dist
        if self.gaussian_step:
            gaussian_dist = self.ge.expand(dist)
            graph.edata['gaussian_dist'] = gaussian_dist
        return graph

    def choose_k_nearest_CWC(self, adj, dist, k, strict):
        for row in range(len(adj)):
            if len(dist[row, :]) > k:
                if strict:
                    adj[row, dist[row, :].argsort()[:k]] = 0
                else:
                    threshold = dist[row, dist[row, :].argsort()[k - 1]]
                    adj[row, dist[row, :] > threshold] = 0
        return adj


class GaussianExpansion(object):

    def __init__(self,
                 dmin,
                 dmax,
                 step=0.5,
                 var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        assert dmax < np.inf
        self.center = torch.arange(dmin, dmax + step, step, device=device)
        self.var = step if var is None else var

    def expand(self, dist):
        return torch.exp(-(dist[..., None] - self.center) ** 2 / self.var ** 2)


class SphericalFeatures(torch.nn.Module):
    def __init__(self, num_spherical=3, num_radial=6, cutoff=5, envelope_exponent=5):
        super(SphericalFeatures, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb
