from dataset import Dataloader
from matgnn import MatGNN
from nn_utils.spherenet import SphereNet

dataset_name = 'QM9'
node_fea_sel = ['atomic_number']
edge_fea_sel = []
label_name = 'U0_atom'
cutoff = 5
k = 12
connect_method = 'CWC'
batch_size = 512
train_rate = 0.841
valid_rate = 0.0764
test_rate = 0.0826
resume = False

data = Dataloader(
    dataset_name, label_name, connect_method=connect_method, cutoff=cutoff, verbose=True, force_reload=False,
    save_graphs=True, save_name='', node_fea_sel=node_fea_sel, edge_fea_sel=edge_fea_sel)

train_loader, valid_loader, test_loader = data.get_split_loaders(train_rate, valid_rate, test_rate, batch_size)

model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=1, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True)

nn = MatGNN(dataset_name, model, train_loader, valid_loader, test_loader, resume=resume)

nn.train()
