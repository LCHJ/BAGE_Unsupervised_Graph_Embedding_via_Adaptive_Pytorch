import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
emb_save = path + "\BAGE\Code\embedding\VBAGE"
sys.path.append(path + '\Graph_Embedding\\functions')

from BAGE_model import *
from metrics import *
from data_process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
import warnings
from link_prediction import *

warnings.filterwarnings('ignore')

#  features: X (n × d); adjacency: similarity matrix; labels: Y
#  Parameters that need to be entered manually
################################################ Please Reading !!!!#################################################
# if the Dataset is "cora" / "citeseer", the learning_rate can be set as "5 * 1e-4". which will bring a better results.
# Increasing the learning rate a little can achieve better results in faster iterations

start_time = time.time()
######################################################### Setting #####################################################
dataset = 'citeseer'
CLASSIFICATION = False
CLUSTERING = False
LINKPREDICTION = True
TSNE = True
scale = 0
emb_save = "./embedding/VBAGE/"
########################################## hyper-parameters##############################################################

# pubmed link_prediction
epoch_num = 160
learning_rate = 1e-3
Lambda = 1

# epoch_num = 200
# learning_rate = 5*1e-4
hidden_layer_1 = 1024
hidden_layer_2 = 128

# Lambda = 10
alpha = 0.1

# every q epoch to update the graph
update_gap = 5
#
num_update = 5

################################### Load dataset   ######################################################################
if dataset in ['cora', 'citeseer', 'pubmed']:
    load_data = LoadData(dataset)
    features, labels, adjacency_matrix_raw = load_data.graph()

elif dataset is 'obg-ddi':
    load_data = LoadData(dataset)
    features, adjacency_matrix_raw = load_data.obg_ddi()

else:
    load_data = LoadData(dataset)
    features, labels = load_data.mat()

################################### Calculate the adjacency matrix #########################################################
if 'adjacency_matrix_raw' in vars() or (dataset in ['cora', 'citeseer', 'pubmed']):
    print('adjacency matrix is raw')
    pass
else:
    print('adjacency matrix is caculated by CAN')
    graph = GraphConstruction(features)
    adjacency_matrix_raw = graph.can()

################################### Link Prediction   ##################################################################
if LINKPREDICTION:
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
        sp.coo_matrix(adjacency_matrix_raw))
    adjacency_matrix = adj_train.todense()
    adjacency_matrix = torch.Tensor(adjacency_matrix)
    features = torch.Tensor(features)
else:
    features = torch.Tensor(features)
    adjacency_matrix = torch.Tensor(adjacency_matrix_raw)

########## #####################################layers ###############################################################
input_dim = features.shape[1]

B = adjacency_matrix * (20 - 1) + 1
B = torch.Tensor(B)
############################################# is incomplate ############################################################
if scale == 1:
    adjacency_matrix = torch.Tensor(GraphConstruction(np.array(features)).CAN())
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))
    print("The graph is reonstructed by CAN")
elif scale != 0:
    adjacency_matrix = adjacency_incomplete(adjacency_matrix, scale)
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))
else:
    pass

###############################################  Convolution kernel initialization #####################################
adjacency_matrix_initial = torch.clone(adjacency_matrix)
adjacency_matrix_new = torch.clone(adjacency_matrix)

convolution_kernel = ConvolutionKernel(adjacency_matrix)
adjacency_convolution_kernel = convolution_kernel.adjacency_convolution()
laplacian_convolution_kernel = convolution_kernel.laplacian_convolution()
############################################## Result initialization ####################################################
acc_VBAGE_total = []
nmi_VBAGE_total = []
pur_VBAGE_total = []

acc_VBAGE_total_std = []
nmi_VBAGE_total_std = []
pur_VBAGE_total_std = []

F1_score = []

roc_score = []
ap_score = []

roc_score_std = []
ap_score_std = []
############################################## Model ################################################################
mse_loss = torch.nn.MSELoss(size_average=False)
model_VBAGE = VBAGE(features.shape[1], hidden_layer_1, hidden_layer_2)
optimzer = torch.optim.Adam(model_VBAGE.parameters(), lr=learning_rate, weight_decay=10)
####################################################### GPU ###########################################################
# if torch.cuda.is_available():
#     model_VBAGE = model_VBAGE.cuda(0)
#     features = features.cuda(0)
#     B = B.cuda(0)
#     adjacency_matrix_new = adjacency_matrix_new.cuda(0)
#     adjacency_convolution_kernel = adjacency_convolution_kernel.cuda(0)
#     laplacian_convolution_kernel = laplacian_convolution_kernel.cuda(0)
####################################################### Train the Model ################################################
for epoch in range(epoch_num):
    
    embedding, graph_reconstruction, H_2_mean, H_2_std = model_VBAGE(adjacency_convolution_kernel, features)
    
    # loss_1 = torch.norm((graph_reconstruction - adjacency_matrix_new) * B, p='fro')
    loss_1 = mse_loss((graph_reconstruction - adjacency_matrix_new) * B, torch.zeros_like(adjacency_matrix))
    loss_2 = -0.5 / graph_reconstruction.size(0) * (1 + 2 * H_2_std - H_2_mean ** 2 - torch.exp(H_2_std) ** 2).sum(
        1).mean()
    loss_3 = Lambda * torch.trace((embedding.T).mm(laplacian_convolution_kernel).mm(embedding))
    loss = loss_1 + loss_2 + Lambda * loss_3
    ####################################################  Update the Convolution Kernel ################################
    #   update the graph every 5
    if scale != 0 and epoch > update_gap and epoch % update_gap == 0 and epoch < num_update * update_gap:
        ############################### cahnge to the CPU, If delete the detach, the process will be very slow #########
        embedding = embedding.detach().cpu()
        adjacency_convolution_kernel = adjacency_convolution_kernel.detach()
        laplacian_convolution_kernel = laplacian_convolution_kernel.detach()
        ################################## update the Adjacency_matrix #################################################
        A_update = adjacency_update(embedding, adjacency_matrix_new, is_symmetry=True, is_round=False)
        print('adjacency update success')
        adjacency_matrix_new = alpha * A_update + (1 - alpha) * adjacency_matrix_initial
        convolution_kernel_update = ConvolutionKernel(adjacency_matrix_new)
        adjacency_convolution_kernel = convolution_kernel_update.adjacency_convolution()
        laplacian_convolution_kernel = convolution_kernel_update.laplacian_convolution()
    
    ##################################################### restore to the GPU ###########################################
    # adjacency_matrix_new = adjacency_matrix_new.cuda(0)
    # embedding = embedding.cuda(0)
    # adjacency_convolution_kernel = adjacency_convolution_kernel.cuda(0)
    # laplacian_convolution_kernel = laplacian_convolution_kernel.cuda(0)
    
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    
    embedding = embedding.cpu().detach().numpy()
    #############################  Calculate the clustering result #####################################################
    
    if CLASSIFICATION and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = my_SVM(embedding, labels, 0.3)
        print("Epoch[{}/{}], F1-score = {}".format(epoch + 1, epoch_num, score))
        F1_score.append(score)
        print("Epoch:{}, Loss_1:{:.4f}, Loss_2:{:.4f}, Loss_3:{:.4f}".format(epoch + 1, loss_1.item(), loss_2.item(),
                                                                             loss_3.item()))
    
    if CLUSTERING and (epoch + 1) % 5 == 0:
        print("Epoch:{}, Loss_1:{:.4f}, Loss_2:{:.4f}, Loss_3:{:.4f}".format(epoch + 1, loss_1.item(), loss_2.item(),
                                                                             Lambda * loss_3.item()))
        
        acc_H2 = []
        nmi_H2 = []
        pur_H2 = []
        
        kmeans = KMeans(n_clusters=max(np.int_(labels).flatten()))
        for i in range(10):
            Y_pred_OK = kmeans.fit_predict(embedding)
            Y_pred_OK = np.array(Y_pred_OK)
            Labels_K = np.array(labels).flatten()
            AM = clustering_metrics(Y_pred_OK, Labels_K)
            acc, nmi, pur = AM.evaluationClusterModelFromLabel(print_msg=False)
            acc_H2.append(acc)
            nmi_H2.append(nmi)
            pur_H2.append(pur)
        
        print(f'ACC_VBAGE=', 100 * np.mean(acc_H2), '\n', 'NMI_VBAGE=', 100 * np.mean(nmi_H2), '\n', 'PUR_VBAGE=',
              100 * np.mean(pur_H2))
        
        acc_VBAGE_total.append(100 * np.mean(acc_H2))
        nmi_VBAGE_total.append(100 * np.mean(nmi_H2))
        pur_VBAGE_total.append(100 * np.mean(pur_H2))
        
        acc_VBAGE_total_std.append(100 * np.std(acc_H2))
        nmi_VBAGE_total_std.append(100 * np.std(nmi_H2))
        pur_VBAGE_total_std.append(100 * np.std(pur_H2))
        np.save(emb_save + "{}.npy".format(epoch + 1), embedding)
    
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

################################################ Results ###############################################################
if CLUSTERING:
    index_max = np.argmax(acc_VBAGE_total)
    
    print('ACC_VBAGE_max={:.2f} +- {:.2f}'.format(np.float(acc_VBAGE_total[index_max]), \
                                                  np.float(acc_VBAGE_total_std[index_max])))
    
    print('NMI_VBAGE_max={:.2f} +- {:.2f}'.format(np.float(nmi_VBAGE_total[index_max]), \
                                                  np.float(nmi_VBAGE_total_std[index_max])))
    
    print('PUR_VBAGE_max={:.2f} +- {:.2f}'.format(np.float(pur_VBAGE_total[index_max]), \
                                                  np.float(pur_VBAGE_total_std[index_max])))
    print("The incompleteness of the adjacency matrix is {}%".format(scale * 100))

if CLASSIFICATION:
    index_max = np.argmax(F1_score)
    print("VBAGE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))

if LINKPREDICTION:
    index_max_roc = np.argmax(roc_score)
    index_max_ap = np.argmax(ap_score)
    
    print("VBAGE: AUC_max is {:.2f} +- {:.4f}".format(100 * np.float(roc_score[index_max_roc]), \
                                                      100 * np.float(roc_score_std[index_max_roc])))
    
    print("VBAGE: AP_max is {:.2f} +- {:.4f}".format(100 * np.float(ap_score[index_max_ap]), \
                                                     100 * np.float(ap_score_std[index_max_ap])))

if TSNE:
    index_max = index_max_roc
    print("dataset is {}".format(dataset))
    print("Index_Max = {}".format(index_max))
    latent_representation_max = np.load(emb_save + "{}.npy".format((index_max + 1) * 5))
    features = features.cpu().numpy()
    plot_embeddings(latent_representation_max, features, labels)

########################################################################################################################
end_time = time.time()
print("Running time is {}".format(end_time - start_time))
