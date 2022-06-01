# -*- coding:utf-8 -*-
# @Time : 2022/5/17 9:40
# @Author: LCHJ
# @File : main.py.py
import sys
sys.path.append('./..')
from Graph_Embedding.functions.data_process import adjacency_incomplete

import argparse
import time
import warnings

import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_sparse import SparseTensor

from BAGE_Code.SparseBAGE_model import SparseBAGE, loss_Laplacian, recon_loss, gen_neg_edge_index
from BAGE_Code.get_laplacian import get_laplacian
from BAGE_model import *
from sklearn import svm
# from thundersvm import SVC
warnings.filterwarnings('ignore')


# ------------------------ Setting ------------------------
#  Parameters that need to be entered manually
def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser(description='SparseBAGE: L1 + λL2, SVC ,(ogbn-arxiv)')
    parser.add_argument('--dataset', type=str, default="ogbn-arxiv")  # ogbn-products ogbn-arxiv

    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--is_softmax', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=40)

    parser.add_argument('--k_loss0', type=int, default=8)
    parser.add_argument('--num_pos', type=int, default=4)
    parser.add_argument('--num_neg', type=int, default=2)
    parser.add_argument('--lambda_', type=float, default=1e-3, help="L = L1 + λL2")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="A = α A_L + (1 - α) A_0, balance the trade-off between the learned A_L and A_0")
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.99)

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--version', type=str, default="1")
    parser.add_argument('--runs', type=int, default=10)

    return parser.parse_args()


# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
# if_cuda = False

args = parse_args()
print(f"===>>> 0. Completing Setting Parameters! args: <<<===\n--->>> {args} --->>>")
print(f'===>>> [SparseBAGE_V.{args.version}] | dataset: {args.dataset} |'
      f'[loss = {args.k_loss0}* loss0 + [{args.num_pos}*pos + {args.num_neg}*neg] + {args.lambda_} * loss_2] |if_cuda= {if_cuda}')
# ------------------------ hyper-parameters ------------------------
# 查看当前 gpu id
device = f'cuda:{args.device}' if if_cuda else 'cpu'
device = torch.device(device)

CLASSIFICITION = True
CLUSTERING = False
TSNE = False
LINKPREDICTION = False
scale = 0
emb_save = "./embedding/BAGE/"
model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './model/'))
# pubmed in link_prediction task with bigger learning_rate
# learning_rate = 5 * 1e-4
update_gap = 5  # update the graph structure every q epochs.
num_update = 5  # set a threshold τ for stopping updates


# ------------------------  1. Load dataset ------------------------
def LoadData_Sparse(dataname):
    dataset = PygNodePropPredDataset(name=dataname)
    data = dataset[0]
    # 3 edge_index  adjacency_matrix_raw 边的坐标index # torch.Size([2, num])
    edge_index = sort_edge_index(data.edge_index).to(device)
    self_loop = False
    if self_loop:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    transform = T.ToSparseTensor()
    data.adj_t = transform(data).adj_t.to_symmetric().to(device)
    split_idx = dataset.get_idx_split()
    print(
        '--->>> adjacency matrix is adj_t.to_symmetric of torch.sparse with self_loop = {} \t--->>>'.format(self_loop))
    return data, edge_index.to(device), split_idx


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator, is_softmax=True):
    model.eval()
    pred, z = model(x, edge_index)
    if is_softmax:
        y_pred = pred.argmax(dim=-1, keepdim=True)
        # y_pred = pred.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']], 'y_pred': y_pred[split_idx['train']], })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']], 'y_pred': y_pred[split_idx['valid']], })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']], 'y_pred': y_pred[split_idx['test']], })[
            'acc']  # f1 = f1_score(y_true[split_idx['test']].cpu().detach().numpy(),  #               y_pred[split_idx['test']].cpu().detach().numpy(), average='weighted')
    else:
        z = z.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        skip = 4
        num_samples = 12000
        train_idx = split_idx['train'][::skip][:num_samples]
        clf = svm.SVC(C=0.5, tol=1e-4, cache_size=1900)  # 创建分类器对象
        # clf = SVC(cache_size=8192)
        clf.fit(z[train_idx], y_true[train_idx])

        y_pred = np.array(clf.predict(z[train_idx])).reshape(-1, 1)
        train_acc = evaluator.eval({
            'y_true': y_true[train_idx], 'y_pred': y_pred, })['acc']  # 11368
        valid_idx = split_idx['valid'][::skip][:num_samples]
        y_pred = np.array(clf.predict(z[valid_idx])).reshape(-1, 1)
        valid_acc = evaluator.eval({
            'y_true': y_true[valid_idx], 'y_pred': y_pred, })['acc']  # 3725

        test_idx = split_idx['test'][::skip][:num_samples]
        y_pred = np.array(clf.predict(z[test_idx])).reshape(-1, 1)  # 6079
        test_acc = evaluator.eval({
            'y_true': y_true[test_idx], 'y_pred': y_pred, })[
            'acc']  # f1 = f1_score(y_true[test_idx], y_pred, average='weighted')

    return train_acc, valid_acc, test_acc


def main():
    # ------------------------ Result initialization ------------------------
    acc_BAGE_total = []
    nmi_BAGE_total = []
    pur_BAGE_total = []

    acc_BAGE_total_std = []
    nmi_BAGE_total_std = []
    pur_BAGE_total_std = []

    roc_score = []
    ap_score = []

    roc_score_std = []
    ap_score_std = []
    Best_ACC = np.zeros(3)
    data, edge_index, split_idx = LoadData_Sparse(args.dataset)
    #  Features: X (n × d); adjacency: similarity matrix; labels: Y
    features, labels, adjacency_matrix, num_nodes = data.x.to(device), data.y.to(device), data.adj_t.to(
        device), data.num_nodes
    # if args.dataset in ['cora', 'citeseer', 'pubmed']:
    #     load_data = LoadData('citeseer')
    #     features, labels, adjacency_matrix_raw = load_data.graph()
    #
    #     features = torch.tensor(features, dtype=torch.float32, device=device)
    #     labels = torch.tensor(labels, device=device)
    #     adjacency_matrix = SparseTensor.from_scipy(adjacency_matrix_raw).to(device)
    #     row, col, value = adjacency_matrix.coo()
    #     edge_index = torch.stack([row, col], dim=0)
    #     num_nodes = features.shape[0]

    # 训练集节点index
    train_idx = split_idx['train'].to(device)
    evaluator = Evaluator(name=args.dataset)
    print('===>>> 1. Completing Loading Data! \t\t--->>>')

    # ------------------------ is incomplete------------------------
    if scale == 1:
        adjacency_matrix = torch.Tensor(GraphConstruction(np.array(features)).CAN())
        print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))
        print("The graph is reonstructed by CAN")
    elif scale != 0:
        adjacency_matrix = adjacency_incomplete(adjacency_matrix, scale)
        print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))
    else:
        pass

    # ------------------------ Convolution kernel initialization ------------------------
    # laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index=edge_index, normalization="sym")
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index=edge_index, num_nodes=num_nodes,
                                                                normalization="sym")
    laplacian_convolution_kernel = SparseTensor(row=laplacian_edge_index[0], col=laplacian_edge_index[1],
                                                value=laplacian_edge_weight, sparse_sizes=(num_nodes, num_nodes))
    print('===>>> 2. Completing laplacian_convolution_kernel initialization! \t\t--->>>')
    # ------------------------ Weight Matrix B ------------------------
    # B = adjacency_matrix * 32 + 1
    # ------------------------ Loss Function and GPU------------------------

    model_BAGE = SparseBAGE(features.shape[1], args.hidden_channels, args.out_channels, num_layers=args.num_layers,
                            dropout=args.dropout, num_classes=args.num_classes, isSoftmax=args.is_softmax).to(device)
    model_BAGE.reset_parameters()
    # ν to prevent overfitting,
    optimizer = torch.optim.Adam(model_BAGE.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma, last_epoch=-1)

    # ------------------------ Train the Model ------------------------
    start_time = time.time()
    print('===>>> 3. Starting Training the Model! \t\t--->>>')
    neg_edge_index = gen_neg_edge_index(edge_index, num_nodes, num_neg=args.num_neg)
    for epoch in range(args.epochs + 1):
        # torch.cuda.empty_cache()
        is_curr_great = False
        model_BAGE.train()

        optimizer.zero_grad()
        target, z = model_BAGE(features, edge_index)

        if args.is_softmax:
            loss_0 = args.k_loss0 * F.nll_loss(target[train_idx], labels[train_idx].squeeze(1))
            loss_1 = recon_loss(z, edge_index, neg_edge_index, num_pos=args.num_pos)
            loss_2 = args.lambda_ * (loss_Laplacian(z, laplacian_convolution_kernel) / num_nodes)
            loss = loss_0 + loss_1 + loss_2
        else:
            loss_1 = recon_loss(z, edge_index, neg_edge_index, num_pos=args.num_pos)
            loss_2 = args.lambda_ * (loss_Laplacian(z, laplacian_convolution_kernel) / num_nodes)
            loss_0 = loss_2
            loss = loss_1 + loss_2
        # loss = loss_1 + args.lambda_ * loss_2
        # ------------------------ Update the Convolution Kernel ------------------------ if scale != 0 and epoch >
        # update_gap and epoch % update_gap == 0 and epoch < num_update * update_gap: # cahnge to the CPU,
        # Add detach(), requires_grad = False; If delete the detach, the process will be very slow z = z.detach(
        # ).cpu() adjacency_convolution_kernel = adjacency_convolution_kernel.detach() laplacian_convolution_kernel =
        # laplacian_convolution_kernel.detach() # ------------------------ update the Adjacency_matrix
        # ------------------------ A_update = adjacency_update(z, adjacency_matrix, M_min=3, M_max=50,
        # is_symmetry=True, is_round=True) print('adjacency update success\n') adjacency_matrix_new = args.alpha *
        # A_update + (1 - args.alpha) * adjacency_matrix # adjacency_matrix_new = args.alpha * A_update + (1 -
        # args.alpha) * adjacency_matrix convolution_kernel_update = ConvolutionKernel(adjacency_matrix_new)
        # adjacency_convolution_kernel = convolution_kernel_update.adjacency_convolution()
        # laplacian_convolution_kernel = convolution_kernel_update.laplacian_convolution()
        #
        #     # ------------------------ restore to the GPU ------------------------
        #     adjacency_matrix_new = adjacency_matrix_new.to(device)
        #     z = z.to(device)
        #     adjacency_convolution_kernel = adjacency_convolution_kernel.to(device)
        #     laplacian_convolution_kernel = laplacian_convolution_kernel.to(device)
        #

        loss.backward()
        optimizer.step()
        time_elapsed = time.time() - start_time
        ET = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
        # ------------------------ Calculate the clustering result ------------------------
        if CLASSIFICITION and epoch % 10 == 0:  # epoch > 1200 and
            curr_acc = test(model_BAGE, features, edge_index, labels, split_idx, evaluator, is_softmax=args.is_softmax)
            for i in range(0, len(Best_ACC)):
                if Best_ACC[i] < curr_acc[i]:
                    Best_ACC[i] = curr_acc[i]
                    is_curr_great = True

            if is_curr_great:
                torch.save(model_BAGE, f"{model_path}/Curr_Best_model_{args.version}.pt")
            curr_acc = [float("{:.5f}".format(i)) for i in 100 * np.array(curr_acc)]
            print(f"===>>> Epoch[{epoch + 1}/{args.epochs}]\t[Train-Valid-Test-F1_score]\t= {curr_acc}\t"
                  f"lr=[{scheduler.get_lr()[0]:.6f}]\t"
                  f"[Loss= {loss.item():.4f}  Loss_1= {loss_1.item():.4f}  Loss_2= {loss_2.item():.4f}]\tET= {ET}")  # neg_edge_index = gen_neg_edge_index(edge_index, num_nodes, num_neg=args.num_neg)
        elif epoch % 5 == 0:
            index_max = [float("{:.5f}".format(i)) for i in 100 * Best_ACC]
            print(f"--->>> Epoch[{epoch + 1}/{args.epochs}]\t|-->>>Current Best ACC -->|\t= {index_max}\t"
                  f"lr=[{scheduler.get_lr()[0]:.6f}]\t"
                  f"[Loss= {loss.item():.4f}  Loss_1= {loss_1.item():.4f}  Loss_0= {loss_0.item():.4f}]\tET= {ET}")
            scheduler.step()

    if CLASSIFICITION:
        index_max = [float("{:.5f}".format(i)) for i in 100 * Best_ACC]
        print(f"--->>>BAGE: Best Result is | Train-Valid-Test-F1_score| \t= {index_max}")

    time_elapsed = time.time() - start_time
    ET = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    print("===>>> Running time is {} <<<===".format(ET))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"--- num_pos num_neg lambda_   weight_decay  hidden_channels	out_channels    num_layers  dropout lr ---\n"
          f"---[{args.num_pos}  {args.num_neg}   {args.lambda_}   {args.weight_decay} {args.hidden_channels}  {args.out_channels} {args.num_layers}   {args.dropout}  {args.lr}]---")
    main()
    print(f"--- num_pos num_neg lambda_   weight_decay  hidden_channels	out_channels    num_layers  dropout lr ---\n"
          f"---[{args.num_pos}  {args.num_neg}   {args.lambda_}   {args.weight_decay} {args.hidden_channels}  {args.out_channels} {args.num_layers}   {args.dropout}  {args.lr}]---")
