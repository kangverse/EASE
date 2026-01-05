## EASE

We propose a novel Uncertainty-Calibrated Elastic Alignment framework, named EASE. 
We reformulate feature imputation as a probabilistic conditional density estimation problem, explicitly capturing the uncertainty inherent in ambiguous cross-modal mappings. 
Building on this, we introduce an uncertainty-adaptive elastic kernel in the Reproducing Kernel Hilbert Space (RKHS), which leverages the estimated confidence to modulate alignment strength and mitigate noise overfitting. 
Finally, we bridge the disconnect at the decision level by imposing consistency constraints between partial and full views, ensuring robust discrimination under varying missingness patterns.



## Usage

### Prerequisites
- Python 3.11.9
- CUDA 12.4
- pytorch ==2.4.0
- torchvision == 0.19.0

(see `requirements.txt` for more details)

### Dataset
We use MOSI, MOSEI and SIMS for evaluation.

- **MOSI**  
  CMU Multimodal Opinion Sentiment Intensity (MOSI) is a benchmark dataset for multimodal sentiment analysis, consisting of short opinion video clips annotated with sentiment intensity labels. Each sample provides aligned text, visual, and acoustic modalities.

- **MOSEI**  
  CMU Multimodal Opinion, Sentiment, and Emotion Intensity (MOSEI) is a large-scale extension of MOSI, covering more diverse topics and speakers. It includes multimodal inputs with sentiment intensity and emotion annotations, and is widely used for evaluating robustness under missing or noisy modalities.

- **CH-SIMS**  
  CH-SIMS is a Chinese multimodal sentiment analysis dataset containing video clips with synchronized Chinese text, visual, and acoustic features. It supports sentiment classification and regression tasks, enabling evaluation of multilingual and cross-cultural MSA models.

> All datasets are publicly available. We use their official or commonly adopted preprocessed features to ensure fair comparison across methods.

We list the datasets links .

| Dataset | Task | Download link |
| :------: | :-----: | :------: |
|  MOSI |          Sentiment Analysis        | [link](https://drive.google.com/drive/folders/1FI85tx0YNAq5dc6gNfsw4lLwS4Dy8pZL)|
|  MOSEI|          Sentiment Analysis       | [link](https://drive.google.com/drive/folders/1umLIjIlL8Y1oWYzU2L6UyPTHFQx7RREB)|
|  CH-SIMS  |         Chinese Sentiment Analysis      | [link](https://drive.google.com/drive/folders/1oplJ15kdS_OK0wHXycI8p77jnmwAHhbG)|

Download the three datasets and put them in the `EASE/dataset` folder before running the scripts.


### Traing EASE and Evaluation

Please download the following pretrained models (`bert-base-uncased` and `bert-based-chinese`) and place them under `EASE/prebert/`.
Afterwards, you can train the model using the script below:

```shell
cd EASE
sh train_EASE.sh

```

Evaluate the average performance under different missing-modality scenarios by using `robust_eval.sh`.

```shell
# evaluate the average performance on the MOSEI dataset
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --save_path robust_results/EASE --file_package checkpoint_path --config_file configs/eval_mosei.yaml --key_eval Has0_acc_2


```