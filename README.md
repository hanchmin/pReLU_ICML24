# pReLU_ICML24

This repository has the supplemental code for the paper: 

H. Min and R. Vidal, “Can Implicit Bias Imply Adversarial Robustness?,” in the 41th International Conference on Machine Learning (ICML). [Paper](https://hanchmin.github.io/assets/pdf/MV2024ICML.pdf)

For the experiments (requires [AutoAttack](https://github.com/fra31/auto-attack)), run

 - (Section 5.1) python prelu_f_dist.py --config 'configs/f_dist/icml24_f_dist_main.yaml' --exp_name 'FOLDER_NAME_FOR_RESULTS'
  
 - (Section 5.2.1) python prelu_mnist_main.py --config 'configs/mnist/icml24_mnist_2_classes.yaml' --exp_name 'FOLDER_NAME_FOR_RESULTS'
  
 - (Appendix B) python prelu_mnist_main.py --config 'configs/mnist/icml24_mnist_10_classes.yaml' --exp_name 'FOLDER_NAME_FOR_RESULTS'
  
 - (Section 5.2.2) python prelu_caltech256.py --config 'configs/caltech256/icml24_caltech256_10_classes.yaml' --exp_name 'FOLDER_NAME_FOR_RESULTS'

To run the experiment in Section 5.2.2, please download caltech256 dataset to 'data/caltech256'

For pretrained networks, please see "prelu_pretrained.py"

