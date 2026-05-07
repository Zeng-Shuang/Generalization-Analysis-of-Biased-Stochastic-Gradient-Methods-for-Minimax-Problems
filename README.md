# Generalization-Analysis-of-Biased-Stochastic-Gradient-Methods-for-Minimax-Problems

## Requirements
```
python==3.10
numpy==2.2.5
scikit-learn==1.7.2
matplotlib==3.10.8
pytorch==2.5.1
torchvision==0.20.1
tqdm==4.67.1
```
## Train and Read Results
### AUC Maximization
```
cd ./auc
python auc_code/expp.py
python auc_code/plot.py
python auc_code/plot_gen.py
```
### GAN
```
cd ./gan
python gan_code/exp_cv.py
python auc_code/read_results.py
```
