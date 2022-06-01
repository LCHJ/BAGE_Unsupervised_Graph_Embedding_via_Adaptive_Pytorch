import os
import sys

from scipy.sparse import coo_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from Graph_Embedding.functions.data_process import LoadData, adjacency_incomplete, ConvolutionKernel, my_SVM, \
    plot_embeddings
from Graph_Embedding.functions.link_prediction import get_roc_score, mask_test_edges
from Graph_Embedding.functions.metrics import clustering_metrics

# from sklearn.datasets import make_moons

from BAGE_model import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
import warnings

warnings.filterwarnings('ignore')

#  Features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manually

# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
# if_cuda = False
print("=== if_cuda=", if_cuda)
# GPU 的数量
gpu_count = torch.cuda.device_count()
print("--- gpu_count=", gpu_count)
device = torch.device("cuda:0" if if_cuda else "cpu")
# ------------------------ Setting ------------------------
dataset = "citeseer"
# dataset = "ogbn_arxiv"
print("--- dataset is {}".format(dataset))
CLASSIFICITION = True
CLUSTERING = True
TSNE = False
LINKPREDICTION = True
scale = 0
emb_save = "embedding/BAGE/"
# ------------------------ hyper-parameters ------------------------
epoch_num = 100

# pubmed in link_prediction task with bigger learning_rate
# learning_rate = 5 * 1e-4

learning_rate = 1e-3
gamma = 0.99
hidden_layer_1 = 1024
hidden_layer_2 = 128

lambda_ = 0.1
alpha = 0.1

update_gap = 5
num_update = 5
# ------------------------  Load dataset ------------------------
if dataset in ['cora', 'citeseer', 'pubmed', 'ogbn_arxiv']:
    load_data = LoadData(dataset)
    features, labels, adjacency_matrix_raw = load_data.graph()

elif dataset == 'obg-ddi':
    load_data = LoadData(dataset)
    features, adjacency_matrix_raw = load_data.obg_ddi()

else:
    load_data = LoadData(dataset)
    features, labels = load_data.mat()

# ------------------------ Calculate the adjacency matrix ------------------------
if (dataset in ['cora', 'citeseer', 'pubmed', 'ogbn_arxiv']) or ('adjacency_matrix_raw' in vars()):
    print('adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by CAN')
    graph = GraphConstruction(features)
    adjacency_matrix_raw = graph.can()

# ------------------------ Link Prediction ------------------------
if LINKPREDICTION:
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges( \
        coo_matrix(adjacency_matrix_raw))
    features = torch.Tensor(features).to(device)
    adjacency_matrix = torch.Tensor(adj_train.todense()).to(device)

else:

    features = torch.Tensor(features).to(device)
    adjacency_matrix = torch.Tensor(adjacency_matrix_raw.todense()).to(device)

print("--- Loading adjacency_matrix to device! ---")

# ------------------------ Weight Matrix B ------------------------
B = adjacency_matrix * 36 + 1
# ------------------------ is incomplate ------------------------
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

convolution_kernel = ConvolutionKernel(adjacency_matrix)

adjacency_convolution_kernel = convolution_kernel.adjacency_convolution().to(device)
laplacian_convolution_kernel = convolution_kernel.laplacian_convolution().to(device)

# ------------------------ Result initialization ------------------------
acc_BAGE_total = []
nmi_BAGE_total = []
pur_BAGE_total = []

acc_BAGE_total_std = []
nmi_BAGE_total_std = []
pur_BAGE_total_std = []

F1_score = []

roc_score = []
ap_score = []

roc_score_std = []
ap_score_std = []
# ------------------------ Loss Function and GPU------------------------
mse_loss = torch.nn.MSELoss(reduction='sum')
model_BAGE = BAGE(features.shape[1], hidden_layer_1, hidden_layer_2).to(device)
optimzer = torch.optim.Adam(model_BAGE.parameters(), lr=learning_rate, weight_decay=2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimzer, gamma, last_epoch=-1)

# ------------------------ Train the Model ------------------------
start_time = time.time()
torch.cuda.empty_cache()

for epoch in range(epoch_num):
    # torch.cuda.empty_cache()
    model_BAGE.train()
    graph_reconstruction, embedding = model_BAGE(adjacency_convolution_kernel, features)
    
    # loss_1 = torch.norm((graph_reconstruction - adjacency_matrix) * B, p='fro')
    loss_1 = mse_loss(graph_reconstruction * B, adjacency_matrix * B)
    loss_2 = torch.trace((embedding.T).mm(laplacian_convolution_kernel).mm(embedding))
    loss = loss_1 + lambda_ * loss_2
    # ------------------------------------------------------############  Update the Convolution Kernel ##
    if scale != 0 and epoch > update_gap and epoch % update_gap == 0 and epoch < num_update * update_gap:
        ################ cahnge to the CPU, If delete the detach, the process will be very slow #########
        embedding = embedding.detach().cpu()
        adjacency_convolution_kernel = adjacency_convolution_kernel.detach()
        laplacian_convolution_kernel = laplacian_convolution_kernel.detach()
        #### update the Adjacency_matrix #------------------------------------------------------#########
        A_update = adjacency_update(embedding, adjacency_matrix, M_min=3, M_max=50, is_symmetry=True, is_round=True)
        print('adjacency update success\n')
        adjacency_matrix_new = alpha * A_update + (1 - alpha) * adjacency_matrix
        # adjacency_matrix_new = alpha * A_update + (1 - alpha) * adjacency_matrix
        convolution_kernel_update = ConvolutionKernel(adjacency_matrix_new)
        adjacency_convolution_kernel = convolution_kernel_update.adjacency_convolution()
        laplacian_convolution_kernel = convolution_kernel_update.laplacian_convolution()
        
        ######### restore to the GPU #------------------------------------------------------##############
        adjacency_matrix_new = adjacency_matrix_new.to(device)
        embedding = embedding.to(device)
        adjacency_convolution_kernel = adjacency_convolution_kernel.to(device)
        laplacian_convolution_kernel = laplacian_convolution_kernel.to(device)
    
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    embedding = embedding.cpu().detach().numpy()
    ##############  Calculate the clustering result #------------------------------------------------------#############
    
    if CLASSIFICITION and (epoch + 1) % 5 == 0:
        print("=== Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels)
        print("--- Epoch[{}/{}], F1_score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)
        print("--- Epoch:{}, Loss_1:{:.4f}, Loss_2:{:.4f}".format(epoch + 1, loss_1.item(),
                                                                  loss_2.item()) + '\tlr={:.6f}\t'.format(
            scheduler.get_lr()[0]))
        torch.cuda.empty_cache()
    
    if CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch:{}, Loss_1:{:.4f}, Loss_2:{:.4f}".format(epoch + 1, loss_1.item(), loss_2.item()))
        acc_H2 = []
        nmi_H2 = []
        pur_H2 = []
        
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(embedding)
            Labels_K = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, Labels_K)
            acc, nmi, pur = AM.evaluationClusterModelFromLabel(print_msg=False)
            acc_H2.append(acc)
            nmi_H2.append(nmi)
            pur_H2.append(pur)
        
        print(f'ACC_BAGE=', 100 * np.mean(acc_H2), '\n', 'NMI_BAGE=', 100 * np.mean(nmi_H2), '\n', 'PUR_BAGE=',
              100 * np.mean(pur_H2))
        
        acc_BAGE_total.append(100 * np.mean(acc_H2))
        nmi_BAGE_total.append(100 * np.mean(nmi_H2))
        pur_BAGE_total.append(100 * np.mean(pur_H2))
        
        acc_BAGE_total_std.append(100 * np.std(acc_H2))
        nmi_BAGE_total_std.append(100 * np.std(nmi_H2))
        pur_BAGE_total_std.append(100 * np.std(pur_H2))
        
        # np.save(path_result + "{}.npy".format(epoch + 1), embeddding)
    
    if LINKPREDICTION and (epoch + 1) % 5 == 0:
        roc_i = []
        ap_i = []
        for i in range(10):
            roc_score_temp, ap_score_temp = get_roc_score(test_edges, test_edges_false, embedding)
            roc_i.append(roc_score_temp)
            ap_i.append(ap_score_temp)
        
        roc_score.append(np.mean(roc_i))
        roc_score_std.append(np.std(roc_i))
        
        ap_score.append(np.mean(ap_i))
        ap_score_std.append(np.std(ap_i))
        
        print("Epoch: [{}]/[{}]".format(epoch + 1, epoch_num))
        print("AUC = {}".format(roc_score_temp))
        print("AP = {}".format(ap_score_temp))
        np.save(emb_save + "{}.npy".format(epoch + 1), embedding)
    
    scheduler.step()
##### Results #------------------------------------------------------######
if CLUSTERING:
    index_max = np.argmax(acc_BAGE_total)
    
    print('ACC_BAGE_max={:.2f} +- {:.2f}'.format(np.float(acc_BAGE_total[index_max]), \
                                                 np.float(acc_BAGE_total_std[index_max])))
    print('NMI_BAGE_max={:.2f} +- {:.2f}'.format(np.float(nmi_BAGE_total[index_max]), \
                                                 np.float(nmi_BAGE_total_std[index_max])))
    print('PUR_BAGE_max={:.2f} +- {:.2f}'.format(np.float(pur_BAGE_total[index_max]), \
                                                 np.float(pur_BAGE_total_std[index_max])))
    
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))

if CLASSIFICITION:
    index_max = np.argmax(F1_score)
    print("BAGE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

if LINKPREDICTION:
    index_max_roc = np.argmax(roc_score)
    index_max_ap = np.argmax(ap_score)
    
    print("BAGE: AUC_max is {:.2f} +- {:.4f}".format(100 * np.float(roc_score[index_max_roc]), \
                                                     100 * np.float(roc_score_std[index_max_roc])))
    
    print("BAGE: AP_max is {:.2f} +- {:.4f}".format(100 * np.float(ap_score[index_max_ap]), \
                                                    100 * np.float(ap_score_std[index_max_ap])))

# ------------------------------------------------------#### t-SNE #------------------------------------------------------#########
if TSNE:
    index_max = index_max_roc
    print("dataset is {}".format(dataset))
    print("Index_Max = {}".format(index_max))
    latent_representation_max = np.load(emb_save + "{}.npy".format((index_max + 1) * 5))
    features = features.cpu().numpy()
    plot_embeddings(latent_representation_max, features, labels)
# ------------------------------------------------------

end_time = time.time()
print("Running time is {}".format(end_time - start_time))
