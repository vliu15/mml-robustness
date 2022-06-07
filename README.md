# Distributionally Robust Multi-Task Networks

We explore the utility of multi-task learning in improving performance of group-robustness approaches. Because all group-robustness approaches only consider one pair of spuriously correlated attributes (`Blond_Hair` x `Male`), the reader should skim this README in order:
- [Single-Task Learning](#single-task-learning)
- [Spurious Correlation Identification](#spurious-correlation-identification)
- [Multi-Task Learning](#multi-task-learning)

## Single-Task Learning (STL)
Currently we only experiment with ResNet-50 on the CelebA dataset with a couple group-robustness methods in addition to the standard Empirical Risk Minimization (ERM) optimization of neural networks. We use the `hydra-core` package for  general command-line control of hyperparameters.

### Empirical Risk Minimization
Run
```bash
python train_erm.py exp=erm
```
where default hyperparameters are specified in `configs/exp/erm.yaml`. Command-line control of config fields can be done with flags like
```bash
python train_erm.py exp=erm exp.optimizer.lr=0.0001 exp.train.total_epochs=50 exp.dataset.groupings=['Blond_Hair:Male']
```
The `exp.dataset.groupings` field manages the task that is being trained with any spurious correlations that should be taken into account seperated from the task by a colon. 

In general, the `model` field should not be changed, the `optimizer` field should only be changed for hyperparameter tuning, and the `dataset`/`dataloader` fields should only be changed for data subsampling/reweighting methods, respectively.

### Just Train Twice
We implement [Just Train Twice](https://arxiv.org/abs/2107.09044) (JTT), a popular group-robustness framework that trains a toy network to rebalance the dataset for training the actual network. Run
```bash
python train_jtt.py exp=jtt
```
> We implement `train_jtt.py` as two subprocess calls to `train_erm.py` with error set construction in between, so running this optimization procedure requires its own script and config format.

where default hyperparameters are specified in `configs/exp/jtt.yaml`, which loads `configs/exp/jtt_stage_1.yaml` and `configs/exp/jtt_stage_2.yaml` as subconfigs for each stage of training, respectively.

### Simple Data Rebalancing
We also implement all 4 methods described in [Simple Data Rebalancing](https://arxiv.org/abs/2110.14503), which shows that simply reweighting or subsampling across class or group labels can close worst-group accuracy gaps significantly.

#### Reweighting
Reweighting refers to sampling examples from the dataloader such that in expectation, all subsets of examples are uniformly represented per batch. Specifically, we implement reweighting by class label (RWY) and reweighting by group label (RWG). These can both be specified from the ERM config as
```bash
# RWY
python train_erm.py exp=erm \
    exp.dataloader.sampler=rwy \
    exp.dataloader.batch_size=4 \
    exp.optimizer.lr=0.00002511886 \
    exp.optimizer.weight_decay=0.03981071705 \
    exp.total_epochs=60

# RWG
python train_erm.py exp=erm \
    exp.dataloader.sampler=rwg \
    exp.dataloader.batch_size=32 \
    exp.optimizer.lr=0.00001 \
    exp.optimizer.weight_decay=0.1 \
    exp.total_epochs=60
```
but for simplicity, we give them their own configs
```bash
# RWY
python train_erm.py exp=rwy

# RWG
python train_erm.py exp=rwg
```
where default hyperparameters for RWY and RWG are specified in `configs/exp/rwy.yaml` and `configs/exp/rwg.yaml`, respectively.

#### Subsampling
Subsampling refers to creating a truncated version of the original dataset such that all subsets of examples have the same size. The neural network is then trained on this truncated dataset where all subsets are equally represented. Specifically, we implement subsampling by class label (SUBY) and subsampling by group label (SUBG). These can both be specified from the ERM config as
```bash
# SUBY
python train_erm.py exp=erm \
    exp.dataloader.batch_size=32 \
    exp.dataset.subsample=true \
    exp.dataset.subsample_type=suby \
    exp.optimizer.lr=0.00003981071 \
    exp.optimizer.weight_decay=0.06309573444 \
    exp.total_epochs=60

# SUBG
python train_erm.py exp=erm \
    exp.dataloader.batch_size=4 \
    exp.dataset.subsample=true \
    exp.dataset.subsample_type=subg \
    exp.optimizer.lr=0.00006309573 \
    exp.optimizer.weight_decay=0.01 \
    exp.total_epochs=60
```
but for simplicity, we give them their own configs
```bash
# SUBY
python train_erm.py exp=suby

# SUBG
python train_erm.py exp=subg
```
where default hyperparameters for SUBY and SUBG are specified in `configs/exp/suby.yaml` and `configs/exp/subg.yaml`, respectively.

## Spurious Correlation Identification
Multi-task learning requires additional groupings of spurious correlations in addition to the default `Blond_Hair` x `Male` studied in literature. Therefore, we provide a comprehensive set of scripts to
1. Tune and train ERM neural networks on all 40 attributes of CelebA,
2. Evaluate them on all attribute pairs and measure worst-group performance gaps, and
3. Exhaustively extract all pairs of spurious correlations

In all large-scale scripts that require GPU workload asynchronously across tasks, we provide parallel `sbatch` job submission options in addition to sequential `shell` calls. To adjust the job submission type, append `--mode sbatch` or `--mode shell` to the command (which will default to printing out the commands as `--mode debug`). Our `JobManager` can be found in `scripts/job_manager.py`.

### Tuning and Training
Because tuning on hyperparameter grids on all tasks is very expensive, we instead tune on 5 tasks and apply these hyperparameters to neural networks for all tasks:
```bash
python -m scripts.run_hparam_grid_search --opt erm
```
> These runs will get saved to `./logs`

From these 60 experiments, we find that `lr=0.0001` with `weight_decay=0.1` works the best. Then train 40 ERM neural networks for 25 epochs, one for each task:
```bash
python -m scripts.run_spurious_train --lr 0.0001 --wd 0.1 --batch_size 128 --epochs 25 --mode sbatch
```
> These runs will get saved to`./logs/spurious_id`

### Evaluation
After getting all 40 ERM neural networks, one for each task, we evaluate them on groups created with respect to all 40 attributes to measure each worst-group performance:
```bash
python -m scripts.run_create_spurious_matrix --meta_log_dir ./logs/spurious_id --json_dir ./outputs/spurious_eval --mode sbatch
```
> The outputs for each task will get saved to `./outputs/spurious_eval/$TASK`

For each task, this will call `test.py` to run inference on each `$TASK` x `$ATTR` pair and then aggregate all 40 into heatmaps showing group performances, group sizes, and our spurious correlation delta metric: `delta = |g0 + g3 - g1 - g2|`.

### Spurious Correlation Extraction (EXPERIMENTAL)
Finally, we take all 40x39 possible pairs of spurious correlations (namely, their `delta` metrics) and systematically identify which attributes are spuriously correlated with which tasks. We observe that attributes aren't IID - specifically, there are a lot of labelled attributes that are correlated with gender. Therefore, we use biclustering and SVD-based methods to extract correlations across/between attributes/tasks:
```bash
python -m scripts.spurious_eval --json_dir outputs/spurious_eval --out_dir outputs/svd --gamma 0.9 --k 10
```
> `--gamma` sets the threshold of variance, `gamma`, that the `d` largest singular values must account for, used to reduce the feature dimension

> `--k` sets the number clusters that biclustering clustering should identify

> The outputs from this script will get saved to `./outputs/svd`

This script runs two methods:
1. Biclustering: The 40x40 `T` matrix is column-normalized with softmax to emphasize larger values of `delta`, which is then put through `SpectralCoclustering`. Various combinations of `T` and `A = transpose(T)` are biclustered.
2. SVD + Clustering: The 40x40 `T` matrix is column-normalized with L2 norm and passed into SVD for feature dimension reduction according to the specified `gamma` value. These are clustered with KMeans
3.Deltas Heatmap: Creates a heatmap of the counts of all delta values across the 40x39 possible pairs. By observing where there is a dropoff in counts for a given value of delta we can determine the threshold for the dataset which which to start identiying which pairs exhibit spurious correlations.

## Multi-Task Learning (MTL)
We implement MTL versions for each of the STL versions specified above. It is generally the same procedure to run a MTL method as it is for the STL version, the only differences that exist are in specifying the multiple tasks to train on as well as the weights for each of the individual tasks. Without loss of generality, we explain everyhting in the context of ERM. 

In order to specify multiple tasks alter the `exp.dataset.groupings` file to contain multiple task, spurious correlation pairs and if you wish for the model to also be evaluated on the subgroups formed by the spurious correlation additionally specifiy that `exp.dataset.subgroup_labels=true`. Below is an ERM model trained for 3 tasks: 

```bash
python train_erm.py exp=erm exp.train.total_epochs=50 exp.dataset.groupings=['Blond_Hair:Male', 'Big_Lips:Chubby', 'Gray_Hair:Young'] exp.dataset.subgroup_labels=true
```

In addition, we can specify how to weight the specified tasks through the following field: `exp.dataset.task_weights`. A list of the same length of the number of tasks should be passed into this field where the value of the index will correspond to the task specified at that index. These weights should either all be 1 or  all sum to 1. Hence, a possible specification for the above scenario is the following: `exp.dataset.task_weights=[0.25, 0.5, 0.25]`

However, note that these weights are all static throughout training and if it is desired to have weights that are dynamic throughout differnet epochs we also support [Loss Balanced Task Weighting](https://ojs.aaai.org//index.php/AAAI/article/view/5125) (LBTW) which updates the weights for a each task based on the loss ratio between the loss of the current batch and the loss of the initial batch. This ratio acts as a metric for how well the model has trained for that given task. LBTW can be specified as follows:

```bash
python train_erm.py exp=erm exp.train.total_epochs=50 exp.dataset.groupings=['Blond_Hair:Male', 'Big_Lips:Chubby', 'Gray_Hair:Young'] exp.dataset.subgroup_labels=true exp.dataset.loss_balanced_task_weighting=true exp.dataset.lbtw_alpha=0.5
```

where alpha is a specific parameter to the LBTW algorithm. When `exp.dataset.loss_balanced_task_weighting` is true, ensure that the weights in `exp.dataset.task_weights` are all 1. 

## MTL Experiments 
Here, we specify the files and commands needed for all the experiments that we ran. We specifically investigated the benefit of MTL over the corresponding STL baseline as well as settings under which MTL performs best. 

### Training
scripts.run_hparam_grid_train -> walk through all opts 

### Evaluation 
scripts.run_hparam_grid_eval -> walk through all opts 

### Results Visualization
give overview of how to use report_results
