import sys
from dataset import dataloader
from matgnn import MatGNN
from nn_utils.spherenet import SphereNet
from nn_utils.dimenetpp import DimeNetPP
from nn_utils.schnet import SchNet
import pandas as pd

if __name__ == "__main__":
    args = sys.argv[1:]
    model_name = args[0]
    target_name = args[1]
    results_dir = args[2]
    batch_size = int(args[3])

    dataset_name = 'MaterialsProject'
    node_fea_sel = []
    edge_fea_sel = []
    label_name = target_name
    cutoff = 6
    connect_method = 'PBC'
    train_rate = 0.6
    valid_rate = 0.2
    test_rate = 0.2
    resume = False
    api_key = '3X2CKEKcGGAJyml3Suu8'

    mpid = pd.read_csv(f'../dataset_utils/dataset_cache/MaterialsProject/mp-ids-46744.csv', header=0,
                       names=['material_id'])
    criterias = []
    for material_id in list(getattr(mpid, 'material_id')):
        criterias.append({"material_id": material_id})

    data = dataloader(
        dataset_name, label_name, connect_method=connect_method, cutoff=cutoff, verbose=True, force_reload=False,
        save_graphs=True, save_name='46744', node_fea_sel=node_fea_sel, edge_fea_sel=edge_fea_sel, criteria=criterias,
        api_key=api_key)

    train_loader, valid_loader, test_loader = data.get_split_loaders(train_rate, valid_rate, test_rate, batch_size,
                                                                     shuffle=True)

    if model_name == 'spherenet':
        model = SphereNet(energy_and_force=False, cutoff=cutoff, num_layers=4,
                          hidden_channels=128, out_channels=1, int_emb_size=64,
                          basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                          num_spherical=3, num_radial=6, envelope_exponent=5,
                          num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True)
    elif model_name == 'dimenetpp':
        model = DimeNetPP(energy_and_force=False, cutoff=cutoff, num_layers=4,
                          hidden_channels=128, out_channels=1, int_emb_size=64,
                          basis_emb_size=8, out_emb_channels=256,
                          num_spherical=3, num_radial=6, envelope_exponent=5,
                          num_before_skip=1, num_after_skip=2, num_output_layers=3)
    elif model_name == 'schnet':
        model = SchNet(energy_and_force=False, cutoff=cutoff, num_layers=4,
                       hidden_channels=256, num_filters=64, num_gaussians=31)
    else:
        raise ValueError('Incorrect model name')

    nn = MatGNN(results_dir, model, train_loader, valid_loader, test_loader, resume=resume)

    nn.train(2000)
