import scipy.io as sio
import torch
import os
import scipy.io as io
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset configuration
dataset_list = ['sdd', 'gsad', 'flowmeter', 'wine', 'magic',
                'shuttle', 'robot_nav', 'wifiloc']
rules_list = [15, 50, 10, 15, 50, 100, 10, 50]
epoch_list = [200, 200, 300, 200, 200, 200, 400, 400]

colors = ["#2ecc71", "#90EE90", "#9b59b6", "#DDA0DD", "#3498db", "#87CEFA", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA", "MLP_BNN", "CNN_BNN", "DGP", "FPN"]

noise_level = ['0.0', '0.1', '0.3', '0.5']

n_dataset = len(dataset_list)
mlp_dropout_tsr = torch.zeros(len(dataset_list), 5, 4)
cnn_dropout_tsr = torch.zeros(len(dataset_list), 5, 4)
mlp_gnia_tsr = torch.zeros(len(dataset_list), 5, 4)
cnn_gnia_tsr = torch.zeros(len(dataset_list), 5, 4)
dgp_tsr = torch.zeros(len(dataset_list), 5, 4)
mlp_bnn_tsr = torch.zeros(len(dataset_list), 5, 4)
cnn_bnn_tsr = torch.zeros(len(dataset_list), 5, 4)
fpn_tsr = torch.zeros(len(dataset_list), 5, 4)

fpn_trend_dict = dict()

for i in torch.arange(n_dataset):
    fpn_trend_tsr = torch.zeros(4, 5, epoch_list[i])
    for j in torch.arange(len(noise_level)):
        load_data_dir = f"{dataset_list[i]}_best_nl_{noise_level[j]}.mat"
        load_path = f"./results/{dataset_list[i]}/{load_data_dir}"
        load_data = sio.loadmat(load_path)
        mlp_dropout = torch.tensor(load_data['mlp_dropout'])
        cnn_dropout = torch.tensor(load_data['cnn_dropout'])
        mlp_gnia = torch.tensor(load_data['mlp_gnia'])
        cnn_gnia = torch.tensor(load_data['cnn_gnia'])
        dgp = torch.tensor(load_data['dgp'])
        fpn = torch.tensor(load_data['fpn'])
        mlp_bnn = torch.tensor(load_data['mlp_bnn'])
        cnn_bnn = torch.tensor(load_data["cnn_bnn"])

        mlp_dropout_final = mlp_dropout[-1, :]
        cnn_dropout_final = cnn_dropout[-1, :]
        mlp_gnia_final = mlp_gnia[-1, :]
        cnn_gnia_final = cnn_gnia[-1, :]
        dgp_final = dgp[-1, :]
        fpn_final = fpn[-1, :]
        mlp_bnn_final = mlp_bnn[-1, :]/100
        cnn_bnn_final = cnn_bnn[-1, :]/100

        mlp_dropout_tsr[i, :, j] = mlp_dropout_final
        cnn_dropout_tsr[i, :, j] = cnn_dropout_final
        mlp_gnia_tsr[i, :, j] = mlp_gnia_final
        cnn_gnia_tsr[i, :, j] = cnn_gnia_final
        dgp_tsr[i, :, j] = dgp_final
        # print(f"{i}&{j}")
        fpn_tsr[i, :, j] = fpn_final
        mlp_bnn_tsr[i, :, j] = mlp_bnn_final
        cnn_bnn_tsr[i, :, j] = cnn_bnn_final
        fpn_trend_tsr[j, :, :] = fpn[0:epoch_list[i], :].t()

    fpn_trend_dict[dataset_list[i]] = fpn_trend_tsr.view(4, -1)
mlp_dropout_tsr = mlp_dropout_tsr.view(n_dataset, -1).numpy()
cnn_dropout_tsr = cnn_dropout_tsr.view(n_dataset, -1).numpy()
mlp_gnia_tsr = mlp_gnia_tsr.view(n_dataset, -1).numpy()
cnn_gnia_tsr = cnn_gnia_tsr.view(n_dataset, -1).numpy()
dgp_tsr = dgp_tsr.view(n_dataset, -1).numpy()
fpn_tsr = fpn_tsr.view(n_dataset, -1).numpy()
mlp_bnn_tsr = mlp_bnn_tsr.view(n_dataset, -1).numpy()
cnn_bnn_tsr = cnn_bnn_tsr.view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 30, 50])
noise = noise_item.repeat([5]).numpy()
sdd_trend_data = []
for i in range(noise.shape[0]):
    sdd_trend_data.append([noise[i], mlp_dropout_tsr[0][i], cnn_dropout_tsr[0][i], mlp_gnia_tsr[0][i],
                           cnn_gnia_tsr[0][i], mlp_bnn_tsr[0][i], cnn_bnn_tsr[0][i],
                           dgp_tsr[0][i], fpn_tsr[0][i]])
sdd_trend = DataFrame(sdd_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
sdd_fpn_trend_data = []
sdd_fpn_trend_tsr = fpn_trend_dict['sdd']
epoch_item = torch.arange(int(sdd_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    sdd_fpn_trend_data.append([epoch_num[i], sdd_fpn_trend_tsr.numpy()[0][i], sdd_fpn_trend_tsr.numpy()[1][i],
                               sdd_fpn_trend_tsr.numpy()[2][i], sdd_fpn_trend_tsr.numpy()[3][i]])
sdd_fpn_trend = DataFrame(sdd_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

gsad_trend_data = []
for i in range(noise.shape[0]):
    gsad_trend_data.append([noise[i], mlp_dropout_tsr[1][i], cnn_dropout_tsr[1][i], mlp_gnia_tsr[1][i],
                           cnn_gnia_tsr[1][i], mlp_bnn_tsr[1][i], cnn_bnn_tsr[1][i],
                           dgp_tsr[1][i], fpn_tsr[1][i]])
gsad_trend = DataFrame(gsad_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
gsad_fpn_trend_data = []
gsad_fpn_trend_tsr = fpn_trend_dict['gsad']
epoch_item = torch.arange(int(gsad_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    gsad_fpn_trend_data.append([epoch_num[i], gsad_fpn_trend_tsr.numpy()[0][i], gsad_fpn_trend_tsr.numpy()[1][i],
                               gsad_fpn_trend_tsr.numpy()[2][i], gsad_fpn_trend_tsr.numpy()[3][i]])
gsad_fpn_trend = DataFrame(gsad_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

flowmeter_trend_data = []
for i in range(noise.shape[0]):
    flowmeter_trend_data.append([noise[i], mlp_dropout_tsr[2][i], cnn_dropout_tsr[2][i], mlp_gnia_tsr[2][i],
                           cnn_gnia_tsr[2][i], mlp_bnn_tsr[2][i], cnn_bnn_tsr[2][i],
                           dgp_tsr[2][i], fpn_tsr[2][i]])
flowmeter_trend = DataFrame(flowmeter_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
flowmeter_fpn_trend_data = []
flowmeter_fpn_trend_tsr = fpn_trend_dict['flowmeter']
epoch_item = torch.arange(int(flowmeter_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    flowmeter_fpn_trend_data.append([epoch_num[i], flowmeter_fpn_trend_tsr.numpy()[0][i], flowmeter_fpn_trend_tsr.numpy()[1][i],
                               flowmeter_fpn_trend_tsr.numpy()[2][i], flowmeter_fpn_trend_tsr.numpy()[3][i]])
flowmeter_fpn_trend = DataFrame(flowmeter_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

wine_trend_data = []
for i in range(noise.shape[0]):
    wine_trend_data.append([noise[i], mlp_dropout_tsr[3][i], cnn_dropout_tsr[3][i], mlp_gnia_tsr[3][i],
                           cnn_gnia_tsr[3][i], mlp_bnn_tsr[3][i], cnn_bnn_tsr[3][i],
                           dgp_tsr[3][i], fpn_tsr[3][i]])
wine_trend = DataFrame(wine_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
wine_fpn_trend_data = []
wine_fpn_trend_tsr = fpn_trend_dict['wine']
epoch_item = torch.arange(int(wine_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    wine_fpn_trend_data.append([epoch_num[i], wine_fpn_trend_tsr.numpy()[0][i], wine_fpn_trend_tsr.numpy()[1][i],
                               wine_fpn_trend_tsr.numpy()[2][i], wine_fpn_trend_tsr.numpy()[3][i]])
wine_fpn_trend = DataFrame(wine_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

magic_trend_data = []
for i in range(noise.shape[0]):
    magic_trend_data.append([noise[i], mlp_dropout_tsr[4][i], cnn_dropout_tsr[4][i], mlp_gnia_tsr[4][i],
                           cnn_gnia_tsr[4][i], mlp_bnn_tsr[4][i], cnn_bnn_tsr[4][i],
                           dgp_tsr[4][i], fpn_tsr[4][i]])
magic_trend = DataFrame(magic_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
magic_fpn_trend_data = []
magic_fpn_trend_tsr = fpn_trend_dict['magic']
epoch_item = torch.arange(int(magic_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    magic_fpn_trend_data.append([epoch_num[i], magic_fpn_trend_tsr.numpy()[0][i], magic_fpn_trend_tsr.numpy()[1][i],
                               magic_fpn_trend_tsr.numpy()[2][i], magic_fpn_trend_tsr.numpy()[3][i]])
magic_fpn_trend = DataFrame(magic_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

shuttle_trend_data = []
for i in range(noise.shape[0]):
    shuttle_trend_data.append([noise[i], mlp_dropout_tsr[5][i], cnn_dropout_tsr[5][i], mlp_gnia_tsr[5][i],
                           cnn_gnia_tsr[5][i], mlp_bnn_tsr[5][i], cnn_bnn_tsr[5][i],
                           dgp_tsr[5][i], fpn_tsr[5][i]])
shuttle_trend = DataFrame(shuttle_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
shuttle_fpn_trend_data = []
shuttle_fpn_trend_tsr = fpn_trend_dict['shuttle']
epoch_item = torch.arange(int(shuttle_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    shuttle_fpn_trend_data.append([epoch_num[i], shuttle_fpn_trend_tsr.numpy()[0][i], shuttle_fpn_trend_tsr.numpy()[1][i],
                               shuttle_fpn_trend_tsr.numpy()[2][i], shuttle_fpn_trend_tsr.numpy()[3][i]])
shuttle_fpn_trend = DataFrame(shuttle_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

robot_trend_data = []
for i in range(noise.shape[0]):
    robot_trend_data.append([noise[i], mlp_dropout_tsr[6][i], cnn_dropout_tsr[6][i], mlp_gnia_tsr[6][i],
                           cnn_gnia_tsr[6][i], mlp_bnn_tsr[6][i], cnn_bnn_tsr[6][i],
                           dgp_tsr[6][i], fpn_tsr[6][i]])
robot_trend = DataFrame(robot_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
robot_fpn_trend_data = []
robot_fpn_trend_tsr = fpn_trend_dict['robot_nav']
epoch_item = torch.arange(int(robot_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    robot_fpn_trend_data.append([epoch_num[i], robot_fpn_trend_tsr.numpy()[0][i], robot_fpn_trend_tsr.numpy()[1][i],
                               robot_fpn_trend_tsr.numpy()[2][i], robot_fpn_trend_tsr.numpy()[3][i]])
robot_fpn_trend = DataFrame(robot_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

wifi_trend_data = []
for i in range(noise.shape[0]):
    wifi_trend_data.append([noise[i], mlp_dropout_tsr[7][i], cnn_dropout_tsr[7][i], mlp_gnia_tsr[7][i],
                           cnn_gnia_tsr[7][i], mlp_bnn_tsr[7][i], cnn_bnn_tsr[7][i],
                           dgp_tsr[7][i], fpn_tsr[7][i]])
wifi_trend = DataFrame(wifi_trend_data, columns=["noise_level", "MLP_dropout", 'CNN_dropout', "MLP_GNIA", "CNN_GNIA",
                                               "MLP_BNN", "CNN_BNN", "DGP", "FPN"])
wifi_fpn_trend_data = []
wifi_fpn_trend_tsr = fpn_trend_dict['wifiloc']
epoch_item = torch.arange(int(wifi_fpn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    wifi_fpn_trend_data.append([epoch_num[i], wifi_fpn_trend_tsr.numpy()[0][i], wifi_fpn_trend_tsr.numpy()[1][i],
                               wifi_fpn_trend_tsr.numpy()[2][i], wifi_fpn_trend_tsr.numpy()[3][i]])
wifi_fpn_trend = DataFrame(wifi_fpn_trend_data, columns=["epoch", "00%", '10%', "30%", "50%"])

# #================plot fpn trend figure====================
# noise_level_list = ["00%", '10%', "30%", "50%"]
# plt.rcParams['figure.figsize']=[16, 8]
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# ax1 = plt.subplot(241)
# ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.35, 1.35), ncol=4, labels=noise_level_list)
#
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=sdd_fpn_trend, color=c, ci='sd', label=a)
# # ax1.set_xlim((0, 51))
# # ax1.set_ylim((0.77, 0.95))
# ax1.set_ylabel('Test Accuracy',fontsize=10)
# ax1.set_xlabel('Epoch',fontsize=10)
# # ax1.locator_params('y',nbins=5)
# # ax1.locator_params('x',nbins=5)
# ax1.legend_.remove()
# ax1.set_title('SDD')
#
# ax2 = plt.subplot(242)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=gsad_fpn_trend, color=c, ci='sd', label=a)
# # ax2.set_xlim((0, 250))
# # ax2.set_ylim((0.77, 0.95))
# ax2.set_ylabel('Test Accuracy',fontsize=10)
# ax2.set_xlabel('Epoch',fontsize=10)
# # ax2.locator_params('y',nbins=5)
# # ax2.locator_params('x',nbins=5)
# ax2.legend_.remove()
# ax2.set_title('GSAD')
#
# ax3 = plt.subplot(243)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=flowmeter_fpn_trend, color=c, ci='sd', label=a)
# # ax3.set_xlim((0, 250))
# # ax3.set_ylim((0.77, 0.95))
# ax3.set_ylabel('Test Accuracy',fontsize=10)
# ax3.set_xlabel('Epoch',fontsize=10)
# # ax3.locator_params('y',nbins=5)
# # ax3.locator_params('x',nbins=5)
# ax3.legend_.remove()
# ax3.set_title('FM')
#
# ax4 = plt.subplot(244)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=wine_fpn_trend, color=c, ci='sd', label=a)
# # ax4.set_xlim((0, 250))
# # ax4.set_ylim((0.77, 0.95))
# ax4.set_ylabel('Test Accuracy',fontsize=10)
# ax4.set_xlabel('Epoch',fontsize=10)
# # ax4.locator_params('y',nbins=5)
# # ax4.locator_params('x',nbins=5)
# ax4.legend_.remove()
# ax4.set_title('WD')
#
# ax5 = plt.subplot(245)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=magic_fpn_trend, color=c, ci='sd', label=a)
# # ax5.set_xlim((0, 250))
# # ax5.set_ylim((0.77, 0.95))
# ax5.set_ylabel('Test Accuracy',fontsize=10)
# ax5.set_xlabel('Epoch',fontsize=10)
# # ax5.locator_params('y',nbins=5)
# # ax5.locator_params('x',nbins=5)
# ax5.legend_.remove()
# ax5.set_title('MGT')
#
# ax6 = plt.subplot(246)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=shuttle_fpn_trend, color=c, ci='sd', label=a)
# # ax6.set_xlim((0, 250))
# # ax6.set_ylim((0.77, 0.95))
# ax6.set_ylabel('Test Accuracy',fontsize=10)
# ax6.set_xlabel('Epoch',fontsize=10)
# # ax6.locator_params('y',nbins=5)
# # ax6.locator_params('x',nbins=5)
# ax6.legend_.remove()
# ax6.set_title('SC')
#
# ax7 = plt.subplot(247)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=robot_fpn_trend, color=c, ci='sd', label=a)
# # ax7.set_xlim((0, 250))
# # ax7.set_ylim((0.77, 0.95))
# ax7.set_ylabel('Test Accuracy',fontsize=10)
# ax7.set_xlabel('Epoch',fontsize=10)
# # ax7.locator_params('y',nbins=5)
# # ax7.locator_params('x',nbins=5)
# ax7.legend_.remove()
# ax7.set_title('WFRN')
#
# ax8 = plt.subplot(248)
# for i in range(len(noise_level_list)):
#     a = noise_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=wifi_fpn_trend, color=c, ci='sd', label=a)
# # ax8.set_xlim((0, 250))
# # ax8.set_ylim((0.77, 0.95))
# ax8.set_ylabel('Test Accuracy',fontsize=10)
# ax8.set_xlabel('Epoch',fontsize=10)
# # ax8.locator_params('y',nbins=5)
# # ax8.locator_params('x',nbins=5)
# ax8.legend_.remove()
# ax8.set_title('WIL')
# plt.savefig(f"./results/fpn_trend.pdf",bbox_inches='tight')

# ================plot fpn trend figure====================
plt.rcParams['figure.figsize']=[16, 8]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.35, 1.35), ncol=8, labels=alg)
tick_label = ['','00%', "10%",'20%','30%','40%','50%']

for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=sdd_trend, color=c, ci='sd', label=a)
# ax1.set_xlim((0, 51))
# ax1.set_ylim((0.77, 0.95))
ax1.set_ylabel('Test Accuracy',fontsize=10)
ax1.set_xlabel('Uncertainty level',fontsize=10)
# ax1.locator_params('y',nbins=5)
# ax1.locator_params('x',nbins=5)
ax1.legend_.remove()
ax1.set_xticklabels(tick_label)
ax1.set_title('SDD')

ax2 = plt.subplot(242)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=gsad_trend, color=c, ci='sd', label=a)
# ax2.set_xlim((0, 250))
# ax2.set_ylim((0.77, 0.95))
ax2.set_ylabel('Test Accuracy',fontsize=10)
ax2.set_xlabel('Uncertainty level',fontsize=10)
# ax2.locator_params('y',nbins=5)
# ax2.locator_params('x',nbins=5)
ax2.legend_.remove()
ax2.set_xticklabels(tick_label)
ax2.set_title('GSAD')

ax3 = plt.subplot(243)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=flowmeter_trend, color=c, ci='sd', label=a)
# ax3.set_xlim((0, 250))
# ax3.set_ylim((0.77, 0.95))
ax3.set_ylabel('Test Accuracy',fontsize=10)
ax3.set_xlabel('Uncertainty level',fontsize=10)
# ax3.locator_params('y',nbins=5)
# ax3.locator_params('x',nbins=5)
ax3.legend_.remove()
ax3.set_xticklabels(tick_label)
ax3.set_title('FM')

ax4 = plt.subplot(244)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=wine_trend, color=c, ci='sd', label=a)
# ax4.set_xlim((0, 250))
# ax4.set_ylim((0.77, 0.95))
ax4.set_ylabel('Test Accuracy',fontsize=10)
ax4.set_xlabel('Uncertainty level',fontsize=10)
# ax4.locator_params('y',nbins=5)
# ax4.locator_params('x',nbins=5)
ax4.legend_.remove()
ax4.set_xticklabels(tick_label)
ax4.set_title('WD')

ax5 = plt.subplot(245)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=magic_trend, color=c, ci='sd', label=a)
# ax5.set_xlim((0, 250))
# ax5.set_ylim((0.77, 0.95))
ax5.set_ylabel('Test Accuracy',fontsize=10)
ax5.set_xlabel('Uncertainty level',fontsize=10)
# ax5.locator_params('y',nbins=5)
# ax5.locator_params('x',nbins=5)
ax5.legend_.remove()
ax5.set_xticklabels(tick_label)
ax5.set_title('MGT')

ax6 = plt.subplot(246)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=shuttle_trend, color=c, ci='sd', label=a)
# ax6.set_xlim((0, 250))
# ax6.set_ylim((0.77, 0.95))
ax6.set_ylabel('Test Accuracy',fontsize=10)
ax6.set_xlabel('Uncertainty level',fontsize=10)
# ax6.locator_params('y',nbins=5)
# ax6.locator_params('x',nbins=5)
ax6.legend_.remove()
ax6.set_xticklabels(tick_label)
ax6.set_title('SC')

ax7 = plt.subplot(247)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=robot_trend, color=c, ci='sd', label=a)
# ax7.set_xlim((0, 250))
# ax7.set_ylim((0.77, 0.95))
ax7.set_ylabel('Test Accuracy',fontsize=10)
ax7.set_xlabel('Uncertainty level',fontsize=10)
# ax7.locator_params('y',nbins=5)
# ax7.locator_params('x',nbins=5)
ax7.legend_.remove()
ax7.set_xticklabels(tick_label)
ax7.set_title('WFRN')

ax8 = plt.subplot(248)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=wifi_trend, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=10)
ax8.set_xlabel('Uncertainty level',fontsize=10)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_xticklabels(tick_label)
ax8.set_title('WIL')
plt.savefig(f"./results/test_trend.pdf",bbox_inches='tight')
print("lslsl")