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
rules_list = [15, 50, 10, 15, 50, 50, 50, 50]
epoch_list = [200, 200, 300, 500, 200, 200, 400, 400]

colors = ["#2ecc71", "#E5E7E9", "#7DCEA0", "#EAF2F8", "#3498db", "#87CEFA",
          "#95a5a6", "#34495E", "#9b59b6", "#D7BDE2", "#e74c3c"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1", "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"]

noise_level = ['0.0', '0.1', '0.3', '0.5']
drop_rate = ['0.05', '0.1', '0.2', '0.3']

n_dataset = len(dataset_list)

cnn11_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
cnn12_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
cnn21_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
cnn22_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)

mlp21_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
mlp121_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
mlp212_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
mlp421_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
mlp12421_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
mlp42124_dropout_tsr = torch.zeros(len(dataset_list), 5, 4, 4)
fpn_tsr = torch.zeros(len(dataset_list), 5, 4, 4)

fpn_dropout_dict = dict()

for i in torch.arange(n_dataset):
    for j in torch.arange(len(noise_level)):
        for k in torch.arange(len(drop_rate)):
            load_data_dir = f"acc_fpn_drop_{dataset_list[i]}_rule{rules_list[i]}_nl_{noise_level[j]}_drop" \
                            f"_{drop_rate[k]}_epoch_{epoch_list[i]}_all.mat"
            load_path = f"./results/{dataset_list[i]}/{load_data_dir}"
            load_data = sio.loadmat(load_path)
            cnn11_dropout = torch.tensor(load_data['cnn11_test_acc_tsr'])
            cnn12_dropout = torch.tensor(load_data['cnn12_test_acc_tsr'])
            cnn21_dropout = torch.tensor(load_data['cnn21_test_acc_tsr'])
            cnn22_dropout = torch.tensor(load_data['cnn22_test_acc_tsr'])

            mlp21_dropout = torch.tensor(load_data['mlp21_test_acc_tsr'])
            mlp121_dropout = torch.tensor(load_data['mlp121_test_acc_tsr'])
            mlp212_dropout = torch.tensor(load_data['mlp212_test_acc_tsr'])
            mlp421_dropout = torch.tensor(load_data['mlp421_test_acc_tsr'])
            mlp12421_dropout = torch.tensor(load_data['mlp12421_test_acc_tsr'])
            mlp42124_dropout = torch.tensor(load_data['mlp42124_test_acc_tsr'])

            load_data_dir = f"{dataset_list[i]}_best_nl_{noise_level[j]}.mat"
            load_path = f"./results/{dataset_list[i]}/{load_data_dir}"
            load_data = sio.loadmat(load_path)
            fpn = torch.tensor(load_data['fpn'])

            cnn11_dropout_final = cnn11_dropout[-1, :]
            cnn12_dropout_final = cnn12_dropout[-1, :]
            cnn21_dropout_final = cnn21_dropout[-1, :]
            cnn22_dropout_final = cnn22_dropout[-1, :]

            mlp21_dropout_final = mlp21_dropout[-1, :]
            mlp121_dropout_final = mlp121_dropout[-1, :]
            mlp212_dropout_final = mlp212_dropout[-1, :]
            mlp421_dropout_final = mlp421_dropout[-1, :]
            mlp12421_dropout_final = mlp12421_dropout[-1, :]
            mlp42124_dropout_final = mlp42124_dropout[-1, :]
            fpn_final = fpn[-1, :]

            cnn11_dropout_tsr[i, :, j, k] = cnn11_dropout_final
            cnn12_dropout_tsr[i, :, j, k] = cnn12_dropout_final
            cnn21_dropout_tsr[i, :, j, k] = cnn21_dropout_final
            cnn22_dropout_tsr[i, :, j, k] = cnn22_dropout_final

            mlp21_dropout_tsr[i, :, j, k] = mlp21_dropout_final
            mlp121_dropout_tsr[i, :, j, k] = mlp121_dropout_final
            mlp212_dropout_tsr[i, :, j, k] = mlp212_dropout_final
            mlp421_dropout_tsr[i, :, j, k] = mlp421_dropout_final
            mlp12421_dropout_tsr[i, :, j, k] = mlp12421_dropout_final
            mlp42124_dropout_tsr[i, :, j, k] = mlp42124_dropout_final
            fpn_tsr[i, :, j, k] = fpn_final

#===============================drop out 0.05================================
drop_rate_idx = 0
cnn11_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn12_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn21_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn22_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

mlp21_dropout_plot = mlp21_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp121_dropout_plot = mlp121_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp212_dropout_plot = mlp212_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp421_dropout_plot = mlp421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp12421_dropout_plot = mlp12421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp42124_dropout_plot = mlp42124_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
fpn_dropout_plot = fpn_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 30, 50])
noise = noise_item.repeat([5]).numpy()
sdd_dropout_data = []
for i in range(noise.shape[0]):
    sdd_dropout_data.append([noise[i], cnn11_dropout_plot[0][i], cnn12_dropout_plot[0][i], cnn21_dropout_plot[0][i],
                           cnn22_dropout_plot[0][i], mlp21_dropout_plot[0][i], mlp121_dropout_plot[0][i],
                           mlp212_dropout_plot[0][i], mlp421_dropout_plot[0][i], mlp12421_dropout_plot[0][i],
                           mlp42124_dropout_plot[0][i], fpn_dropout_plot[0][i]])
sdd_dropout = DataFrame(sdd_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
gsad_dropout_data = []
for i in range(noise.shape[0]):
    gsad_dropout_data.append([noise[i], cnn11_dropout_plot[1][i], cnn12_dropout_plot[1][i], cnn21_dropout_plot[1][i],
                           cnn22_dropout_plot[1][i], mlp21_dropout_plot[1][i], mlp121_dropout_plot[1][i],
                           mlp212_dropout_plot[1][i], mlp421_dropout_plot[1][i], mlp12421_dropout_plot[1][i],
                           mlp42124_dropout_plot[1][i], fpn_dropout_plot[1][i]])
gsad_dropout = DataFrame(gsad_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
flowmeter_dropout_data = []
for i in range(noise.shape[0]):
    flowmeter_dropout_data.append([noise[i], cnn11_dropout_plot[2][i], cnn12_dropout_plot[2][i], cnn21_dropout_plot[2][i],
                           cnn22_dropout_plot[2][i], mlp21_dropout_plot[2][i], mlp121_dropout_plot[2][i],
                           mlp212_dropout_plot[2][i], mlp421_dropout_plot[2][i], mlp12421_dropout_plot[2][i],
                           mlp42124_dropout_plot[2][i], fpn_dropout_plot[2][i]])
flowmeter_dropout = DataFrame(flowmeter_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
wine_dropout_data = []
for i in range(noise.shape[0]):
    wine_dropout_data.append([noise[i], cnn11_dropout_plot[3][i], cnn12_dropout_plot[3][i], cnn21_dropout_plot[3][i],
                           cnn22_dropout_plot[3][i], mlp21_dropout_plot[3][i], mlp121_dropout_plot[3][i],
                           mlp212_dropout_plot[3][i], mlp421_dropout_plot[3][i], mlp12421_dropout_plot[3][i],
                           mlp42124_dropout_plot[3][i], fpn_dropout_plot[3][i]])
wine_dropout = DataFrame(wine_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

magic_dropout_data = []
for i in range(noise.shape[0]):
    magic_dropout_data.append([noise[i], cnn11_dropout_plot[4][i], cnn12_dropout_plot[4][i], cnn21_dropout_plot[4][i],
                           cnn22_dropout_plot[4][i], mlp21_dropout_plot[4][i], mlp121_dropout_plot[4][i],
                           mlp212_dropout_plot[4][i], mlp421_dropout_plot[4][i], mlp12421_dropout_plot[4][i],
                           mlp42124_dropout_plot[4][i], fpn_dropout_plot[4][i]])
magic_dropout = DataFrame(magic_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

shuttle_dropout_data = []
for i in range(noise.shape[0]):
    shuttle_dropout_data.append([noise[i], cnn11_dropout_plot[5][i], cnn12_dropout_plot[5][i], cnn21_dropout_plot[5][i],
                           cnn22_dropout_plot[5][i], mlp21_dropout_plot[5][i], mlp121_dropout_plot[5][i],
                           mlp212_dropout_plot[5][i], mlp421_dropout_plot[5][i], mlp12421_dropout_plot[5][i],
                           mlp42124_dropout_plot[5][i], fpn_dropout_plot[5][i]])
shuttle_dropout = DataFrame(shuttle_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

robot_dropout_data = []
for i in range(noise.shape[0]):
    robot_dropout_data.append([noise[i], cnn11_dropout_plot[6][i], cnn12_dropout_plot[6][i], cnn21_dropout_plot[6][i],
                           cnn22_dropout_plot[6][i], mlp21_dropout_plot[6][i], mlp121_dropout_plot[6][i],
                           mlp212_dropout_plot[6][i], mlp421_dropout_plot[6][i], mlp12421_dropout_plot[6][i],
                           mlp42124_dropout_plot[6][i], fpn_dropout_plot[6][i]])
robot_dropout = DataFrame(robot_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

wifi_dropout_data = []
for i in range(noise.shape[0]):
    wifi_dropout_data.append([noise[i], cnn11_dropout_plot[7][i], cnn12_dropout_plot[7][i], cnn21_dropout_plot[7][i],
                           cnn22_dropout_plot[7][i], mlp21_dropout_plot[7][i], mlp121_dropout_plot[7][i],
                           mlp212_dropout_plot[7][i], mlp421_dropout_plot[7][i], mlp12421_dropout_plot[7][i],
                           mlp42124_dropout_plot[7][i], fpn_dropout_plot[7][i]])
wifi_dropout = DataFrame(wifi_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

# ================plot fpn trend figure====================
plt.rcParams['figure.figsize']=[17, 9]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
tick_label = ['', '00%', "10%", '20%', '30%', '40%', '50%']

for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=sdd_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=gsad_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=flowmeter_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wine_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=magic_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=shuttle_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=robot_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wifi_dropout, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=10)
ax8.set_xlabel('Uncertainty level',fontsize=10)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_xticklabels(tick_label)
ax8.set_title('WIL')
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
plt.savefig(f"./results/dropout{drop_rate[drop_rate_idx]}.pdf",bbox_inches='tight')
plt.close()

#===============================drop out 0.1================================
drop_rate_idx = 1
cnn11_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn12_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn21_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn22_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

mlp21_dropout_plot = mlp21_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp121_dropout_plot = mlp121_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp212_dropout_plot = mlp212_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp421_dropout_plot = mlp421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp12421_dropout_plot = mlp12421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp42124_dropout_plot = mlp42124_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
fpn_dropout_plot = fpn_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 30, 50])
noise = noise_item.repeat([5]).numpy()
sdd_dropout_data = []
for i in range(noise.shape[0]):
    sdd_dropout_data.append([noise[i], cnn11_dropout_plot[0][i], cnn12_dropout_plot[0][i], cnn21_dropout_plot[0][i],
                           cnn22_dropout_plot[0][i], mlp21_dropout_plot[0][i], mlp121_dropout_plot[0][i],
                           mlp212_dropout_plot[0][i], mlp421_dropout_plot[0][i], mlp12421_dropout_plot[0][i],
                           mlp42124_dropout_plot[0][i], fpn_dropout_plot[0][i]])
sdd_dropout = DataFrame(sdd_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
gsad_dropout_data = []
for i in range(noise.shape[0]):
    gsad_dropout_data.append([noise[i], cnn11_dropout_plot[1][i], cnn12_dropout_plot[1][i], cnn21_dropout_plot[1][i],
                           cnn22_dropout_plot[1][i], mlp21_dropout_plot[1][i], mlp121_dropout_plot[1][i],
                           mlp212_dropout_plot[1][i], mlp421_dropout_plot[1][i], mlp12421_dropout_plot[1][i],
                           mlp42124_dropout_plot[1][i], fpn_dropout_plot[1][i]])
gsad_dropout = DataFrame(gsad_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
flowmeter_dropout_data = []
for i in range(noise.shape[0]):
    flowmeter_dropout_data.append([noise[i], cnn11_dropout_plot[2][i], cnn12_dropout_plot[2][i], cnn21_dropout_plot[2][i],
                           cnn22_dropout_plot[2][i], mlp21_dropout_plot[2][i], mlp121_dropout_plot[2][i],
                           mlp212_dropout_plot[2][i], mlp421_dropout_plot[2][i], mlp12421_dropout_plot[2][i],
                           mlp42124_dropout_plot[2][i], fpn_dropout_plot[2][i]])
flowmeter_dropout = DataFrame(flowmeter_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
wine_dropout_data = []
for i in range(noise.shape[0]):
    wine_dropout_data.append([noise[i], cnn11_dropout_plot[3][i], cnn12_dropout_plot[3][i], cnn21_dropout_plot[3][i],
                           cnn22_dropout_plot[3][i], mlp21_dropout_plot[3][i], mlp121_dropout_plot[3][i],
                           mlp212_dropout_plot[3][i], mlp421_dropout_plot[3][i], mlp12421_dropout_plot[3][i],
                           mlp42124_dropout_plot[3][i], fpn_dropout_plot[3][i]])
wine_dropout = DataFrame(wine_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

magic_dropout_data = []
for i in range(noise.shape[0]):
    magic_dropout_data.append([noise[i], cnn11_dropout_plot[4][i], cnn12_dropout_plot[4][i], cnn21_dropout_plot[4][i],
                           cnn22_dropout_plot[4][i], mlp21_dropout_plot[4][i], mlp121_dropout_plot[4][i],
                           mlp212_dropout_plot[4][i], mlp421_dropout_plot[4][i], mlp12421_dropout_plot[4][i],
                           mlp42124_dropout_plot[4][i], fpn_dropout_plot[4][i]])
magic_dropout = DataFrame(magic_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

shuttle_dropout_data = []
for i in range(noise.shape[0]):
    shuttle_dropout_data.append([noise[i], cnn11_dropout_plot[5][i], cnn12_dropout_plot[5][i], cnn21_dropout_plot[5][i],
                           cnn22_dropout_plot[5][i], mlp21_dropout_plot[5][i], mlp121_dropout_plot[5][i],
                           mlp212_dropout_plot[5][i], mlp421_dropout_plot[5][i], mlp12421_dropout_plot[5][i],
                           mlp42124_dropout_plot[5][i], fpn_dropout_plot[5][i]])
shuttle_dropout = DataFrame(shuttle_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

robot_dropout_data = []
for i in range(noise.shape[0]):
    robot_dropout_data.append([noise[i], cnn11_dropout_plot[6][i], cnn12_dropout_plot[6][i], cnn21_dropout_plot[6][i],
                           cnn22_dropout_plot[6][i], mlp21_dropout_plot[6][i], mlp121_dropout_plot[6][i],
                           mlp212_dropout_plot[6][i], mlp421_dropout_plot[6][i], mlp12421_dropout_plot[6][i],
                           mlp42124_dropout_plot[6][i], fpn_dropout_plot[6][i]])
robot_dropout = DataFrame(robot_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

wifi_dropout_data = []
for i in range(noise.shape[0]):
    wifi_dropout_data.append([noise[i], cnn11_dropout_plot[7][i], cnn12_dropout_plot[7][i], cnn21_dropout_plot[7][i],
                           cnn22_dropout_plot[7][i], mlp21_dropout_plot[7][i], mlp121_dropout_plot[7][i],
                           mlp212_dropout_plot[7][i], mlp421_dropout_plot[7][i], mlp12421_dropout_plot[7][i],
                           mlp42124_dropout_plot[7][i], fpn_dropout_plot[7][i]])
wifi_dropout = DataFrame(wifi_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

# ================plot fpn trend figure====================
plt.rcParams['figure.figsize']=[17, 9]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
tick_label = ['','00%', "10%",'20%','30%','40%','50%']

for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=sdd_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=gsad_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=flowmeter_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wine_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=magic_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=shuttle_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=robot_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wifi_dropout, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=10)
ax8.set_xlabel('Uncertainty level',fontsize=10)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_xticklabels(tick_label)
ax8.set_title('WIL')
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
plt.savefig(f"./results/dropout{drop_rate[drop_rate_idx]}.pdf",bbox_inches='tight')
plt.close()

#===============================drop out 0.2================================
drop_rate_idx = 2
cnn11_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn12_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn21_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn22_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

mlp21_dropout_plot = mlp21_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp121_dropout_plot = mlp121_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp212_dropout_plot = mlp212_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp421_dropout_plot = mlp421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp12421_dropout_plot = mlp12421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp42124_dropout_plot = mlp42124_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
fpn_dropout_plot = fpn_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 30, 50])
noise = noise_item.repeat([5]).numpy()
sdd_dropout_data = []
for i in range(noise.shape[0]):
    sdd_dropout_data.append([noise[i], cnn11_dropout_plot[0][i], cnn12_dropout_plot[0][i], cnn21_dropout_plot[0][i],
                           cnn22_dropout_plot[0][i], mlp21_dropout_plot[0][i], mlp121_dropout_plot[0][i],
                           mlp212_dropout_plot[0][i], mlp421_dropout_plot[0][i], mlp12421_dropout_plot[0][i],
                           mlp42124_dropout_plot[0][i], fpn_dropout_plot[0][i]])
sdd_dropout = DataFrame(sdd_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
gsad_dropout_data = []
for i in range(noise.shape[0]):
    gsad_dropout_data.append([noise[i], cnn11_dropout_plot[1][i], cnn12_dropout_plot[1][i], cnn21_dropout_plot[1][i],
                           cnn22_dropout_plot[1][i], mlp21_dropout_plot[1][i], mlp121_dropout_plot[1][i],
                           mlp212_dropout_plot[1][i], mlp421_dropout_plot[1][i], mlp12421_dropout_plot[1][i],
                           mlp42124_dropout_plot[1][i], fpn_dropout_plot[1][i]])
gsad_dropout = DataFrame(gsad_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
flowmeter_dropout_data = []
for i in range(noise.shape[0]):
    flowmeter_dropout_data.append([noise[i], cnn11_dropout_plot[2][i], cnn12_dropout_plot[2][i], cnn21_dropout_plot[2][i],
                           cnn22_dropout_plot[2][i], mlp21_dropout_plot[2][i], mlp121_dropout_plot[2][i],
                           mlp212_dropout_plot[2][i], mlp421_dropout_plot[2][i], mlp12421_dropout_plot[2][i],
                           mlp42124_dropout_plot[2][i], fpn_dropout_plot[2][i]])
flowmeter_dropout = DataFrame(flowmeter_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
wine_dropout_data = []
for i in range(noise.shape[0]):
    wine_dropout_data.append([noise[i], cnn11_dropout_plot[3][i], cnn12_dropout_plot[3][i], cnn21_dropout_plot[3][i],
                           cnn22_dropout_plot[3][i], mlp21_dropout_plot[3][i], mlp121_dropout_plot[3][i],
                           mlp212_dropout_plot[3][i], mlp421_dropout_plot[3][i], mlp12421_dropout_plot[3][i],
                           mlp42124_dropout_plot[3][i], fpn_dropout_plot[3][i]])
wine_dropout = DataFrame(wine_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

magic_dropout_data = []
for i in range(noise.shape[0]):
    magic_dropout_data.append([noise[i], cnn11_dropout_plot[4][i], cnn12_dropout_plot[4][i], cnn21_dropout_plot[4][i],
                           cnn22_dropout_plot[4][i], mlp21_dropout_plot[4][i], mlp121_dropout_plot[4][i],
                           mlp212_dropout_plot[4][i], mlp421_dropout_plot[4][i], mlp12421_dropout_plot[4][i],
                           mlp42124_dropout_plot[4][i], fpn_dropout_plot[4][i]])
magic_dropout = DataFrame(magic_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

shuttle_dropout_data = []
for i in range(noise.shape[0]):
    shuttle_dropout_data.append([noise[i], cnn11_dropout_plot[5][i], cnn12_dropout_plot[5][i], cnn21_dropout_plot[5][i],
                           cnn22_dropout_plot[5][i], mlp21_dropout_plot[5][i], mlp121_dropout_plot[5][i],
                           mlp212_dropout_plot[5][i], mlp421_dropout_plot[5][i], mlp12421_dropout_plot[5][i],
                           mlp42124_dropout_plot[5][i], fpn_dropout_plot[5][i]])
shuttle_dropout = DataFrame(shuttle_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

robot_dropout_data = []
for i in range(noise.shape[0]):
    robot_dropout_data.append([noise[i], cnn11_dropout_plot[6][i], cnn12_dropout_plot[6][i], cnn21_dropout_plot[6][i],
                           cnn22_dropout_plot[6][i], mlp21_dropout_plot[6][i], mlp121_dropout_plot[6][i],
                           mlp212_dropout_plot[6][i], mlp421_dropout_plot[6][i], mlp12421_dropout_plot[6][i],
                           mlp42124_dropout_plot[6][i], fpn_dropout_plot[6][i]])
robot_dropout = DataFrame(robot_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

wifi_dropout_data = []
for i in range(noise.shape[0]):
    wifi_dropout_data.append([noise[i], cnn11_dropout_plot[7][i], cnn12_dropout_plot[7][i], cnn21_dropout_plot[7][i],
                           cnn22_dropout_plot[7][i], mlp21_dropout_plot[7][i], mlp121_dropout_plot[7][i],
                           mlp212_dropout_plot[7][i], mlp421_dropout_plot[7][i], mlp12421_dropout_plot[7][i],
                           mlp42124_dropout_plot[7][i], fpn_dropout_plot[7][i]])
wifi_dropout = DataFrame(wifi_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

# ================plot fpn trend figure====================
plt.rcParams['figure.figsize']=[17, 9]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
tick_label = ['','00%', "10%",'20%','30%','40%','50%']

for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=sdd_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=gsad_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=flowmeter_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wine_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=magic_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=shuttle_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=robot_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wifi_dropout, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=10)
ax8.set_xlabel('Uncertainty level',fontsize=10)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_xticklabels(tick_label)
ax8.set_title('WIL')
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
plt.savefig(f"./results/dropout{drop_rate[drop_rate_idx]}.pdf",bbox_inches='tight')
plt.close()

#===============================drop out 0.3================================
drop_rate_idx = 3
cnn11_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn12_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn21_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
cnn22_dropout_plot = cnn11_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

mlp21_dropout_plot = mlp21_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp121_dropout_plot = mlp121_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp212_dropout_plot = mlp212_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp421_dropout_plot = mlp421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp12421_dropout_plot = mlp12421_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
mlp42124_dropout_plot = mlp42124_dropout_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()
fpn_dropout_plot = fpn_tsr[:, :, :, drop_rate_idx].view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 30, 50])
noise = noise_item.repeat([5]).numpy()
sdd_dropout_data = []
for i in range(noise.shape[0]):
    sdd_dropout_data.append([noise[i], cnn11_dropout_plot[0][i], cnn12_dropout_plot[0][i], cnn21_dropout_plot[0][i],
                           cnn22_dropout_plot[0][i], mlp21_dropout_plot[0][i], mlp121_dropout_plot[0][i],
                           mlp212_dropout_plot[0][i], mlp421_dropout_plot[0][i], mlp12421_dropout_plot[0][i],
                           mlp42124_dropout_plot[0][i], fpn_dropout_plot[0][i]])
sdd_dropout = DataFrame(sdd_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
gsad_dropout_data = []
for i in range(noise.shape[0]):
    gsad_dropout_data.append([noise[i], cnn11_dropout_plot[1][i], cnn12_dropout_plot[1][i], cnn21_dropout_plot[1][i],
                           cnn22_dropout_plot[1][i], mlp21_dropout_plot[1][i], mlp121_dropout_plot[1][i],
                           mlp212_dropout_plot[1][i], mlp421_dropout_plot[1][i], mlp12421_dropout_plot[1][i],
                           mlp42124_dropout_plot[1][i], fpn_dropout_plot[1][i]])
gsad_dropout = DataFrame(gsad_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
flowmeter_dropout_data = []
for i in range(noise.shape[0]):
    flowmeter_dropout_data.append([noise[i], cnn11_dropout_plot[2][i], cnn12_dropout_plot[2][i], cnn21_dropout_plot[2][i],
                           cnn22_dropout_plot[2][i], mlp21_dropout_plot[2][i], mlp121_dropout_plot[2][i],
                           mlp212_dropout_plot[2][i], mlp421_dropout_plot[2][i], mlp12421_dropout_plot[2][i],
                           mlp42124_dropout_plot[2][i], fpn_dropout_plot[2][i]])
flowmeter_dropout = DataFrame(flowmeter_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])
wine_dropout_data = []
for i in range(noise.shape[0]):
    wine_dropout_data.append([noise[i], cnn11_dropout_plot[3][i], cnn12_dropout_plot[3][i], cnn21_dropout_plot[3][i],
                           cnn22_dropout_plot[3][i], mlp21_dropout_plot[3][i], mlp121_dropout_plot[3][i],
                           mlp212_dropout_plot[3][i], mlp421_dropout_plot[3][i], mlp12421_dropout_plot[3][i],
                           mlp42124_dropout_plot[3][i], fpn_dropout_plot[3][i]])
wine_dropout = DataFrame(wine_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

magic_dropout_data = []
for i in range(noise.shape[0]):
    magic_dropout_data.append([noise[i], cnn11_dropout_plot[4][i], cnn12_dropout_plot[4][i], cnn21_dropout_plot[4][i],
                           cnn22_dropout_plot[4][i], mlp21_dropout_plot[4][i], mlp121_dropout_plot[4][i],
                           mlp212_dropout_plot[4][i], mlp421_dropout_plot[4][i], mlp12421_dropout_plot[4][i],
                           mlp42124_dropout_plot[4][i], fpn_dropout_plot[4][i]])
magic_dropout = DataFrame(magic_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

shuttle_dropout_data = []
for i in range(noise.shape[0]):
    shuttle_dropout_data.append([noise[i], cnn11_dropout_plot[5][i], cnn12_dropout_plot[5][i], cnn21_dropout_plot[5][i],
                           cnn22_dropout_plot[5][i], mlp21_dropout_plot[5][i], mlp121_dropout_plot[5][i],
                           mlp212_dropout_plot[5][i], mlp421_dropout_plot[5][i], mlp12421_dropout_plot[5][i],
                           mlp42124_dropout_plot[5][i], fpn_dropout_plot[5][i]])
shuttle_dropout = DataFrame(shuttle_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

robot_dropout_data = []
for i in range(noise.shape[0]):
    robot_dropout_data.append([noise[i], cnn11_dropout_plot[6][i], cnn12_dropout_plot[6][i], cnn21_dropout_plot[6][i],
                           cnn22_dropout_plot[6][i], mlp21_dropout_plot[6][i], mlp121_dropout_plot[6][i],
                           mlp212_dropout_plot[6][i], mlp421_dropout_plot[6][i], mlp12421_dropout_plot[6][i],
                           mlp42124_dropout_plot[6][i], fpn_dropout_plot[6][i]])
robot_dropout = DataFrame(robot_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

wifi_dropout_data = []
for i in range(noise.shape[0]):
    wifi_dropout_data.append([noise[i], cnn11_dropout_plot[7][i], cnn12_dropout_plot[7][i], cnn21_dropout_plot[7][i],
                           cnn22_dropout_plot[7][i], mlp21_dropout_plot[7][i], mlp121_dropout_plot[7][i],
                           mlp212_dropout_plot[7][i], mlp421_dropout_plot[7][i], mlp12421_dropout_plot[7][i],
                           mlp42124_dropout_plot[7][i], fpn_dropout_plot[7][i]])
wifi_dropout = DataFrame(wifi_dropout_data, columns=["noise_level", "CNN_1", "CNN_2", 'CNN_3', "CNN_4", "MLP_1",
                                               "MLP_2", "MLP_3", "MLP_4", "MLP_5", "MLP_6", "FPN"])

# ================plot fpn trend figure====================
plt.rcParams['figure.figsize']=[17, 9]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
tick_label = ['','00%', "10%",'20%','30%','40%','50%']

for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="noise_level", y=a, data=sdd_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=gsad_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=flowmeter_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wine_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=magic_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=shuttle_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=robot_dropout, color=c, ci='sd', label=a)
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
    sns.lineplot(x="noise_level", y=a, data=wifi_dropout, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=10)
ax8.set_xlabel('Uncertainty level',fontsize=10)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_xticklabels(tick_label)
ax8.set_title('WIL')
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.45, 1.39), ncol=8, labels=alg)
plt.savefig(f"./results/dropout{drop_rate[drop_rate_idx]}.pdf",bbox_inches='tight')
plt.close()