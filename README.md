# RFNN
This is the code for the paper "Robust Fuzzy Neural Network with an Adaptive Inference Engine"

# Requirements
python=3.7.13
pytorch~=1.11.0
scipy~=1.7.3
scikit-learn~=1.0.2
pyyaml~=6.0
numpy~=1.21.5
torchmetrics~=0.10.0
torchvision~=0.12.0
matplotlib~=3.5.3
pyro-api~=0.1.2
pyro-ppl~=1.8.2
tqdm~=4.64.1

# Training Settings

Details of each of the models and their corresponding structures and settings follow.
-	Dropout — a training strategy that prevents DNNs from overfitting. This method randomly drops units during training to decrease covariance. We tested both the MLP and CNN variants, denoting the models as MLP-based Dropout and CNN-based Dropout. The dropout rate was chosen from an interval of {0.05,0.1,0.2,0.3}. The best results were regarded as the final performance.
-	GNI — a regularization method that randomly adds Gaussian noise to DNNs to improve robustness. We again tested both MLP and CNN architectures, injecting Gaussian noise into the activation layers in the range of {0.001,0.005,0.01,0.05,0.1,0.3}. The best performance was chosen as the final result.
-	BNN — a combination of Bayesian and DNN methods. Instead of training specific weights to handle noise attacks, this model optimizes the distribution of the DNN weights. With both the MLP and CNN architectures, we estimated the posterior distribution using the no-Uturn sampler [60], which is a self-tuning variant of a Hamiltonian Monte Carlo algorithm [61]. The sampling number for the parameters was set to 100 for all datasets. Additionally, we used warmup during the training progress with a warmup number of 50. Plus, we set the number of discrete steps over which to simulate Hamiltonian dynamics as 40 and the size of the single step taken by the verlet integrator while computing the trajectory using Hamiltonian dynamics as 0.001.
-	GP — a single layer stochastic process that generates the Gaussian distribution of finite input data.
-	DGP — a deep belief network based on a GP algorithm. We tested different numbers of network layers (from 2 to 5), showing the best performance and the final result.
-	FNN — a fuzzy neural network where the firing strengths are calculated by certain exact algorithms (fuzzy AND operations). We varied the number of rules from 2 to 50 and report the best performance.
-	RFNN — our architecture. The variant tested is the basic form of the model. The number of rules K was determined by searching for the best number of FCM clusters from a range of [5 : 5 : 50]. We constructed the inference engine with 2 rules. For the consequent component, we also used a 3-layer MLP to build the defuzzification units.

When training the networks of GNI and Dropout, we adopted cross entropy loss and used the Adam algorithm as the optimizer with a l2 penalty on the model parameters. The learning rate was set to start at 0.1 and decay by 0.1 every 50 epochs with 500 maximum epochs. To avoid wasting computing resources and to save time, we set an early stopping condition to halt if the loss had not decreased for 10 epochs.

When training the GP and DGP models, we used the radial basis function (RBF) as their kernels. Notably, Both BNN, GP, and DGP are implemented via the Python package Pyro

# Run
mAP is obtained via running the code main.py
f1 score is obtained via running the code main_f1.py


