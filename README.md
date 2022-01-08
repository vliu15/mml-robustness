# Distributionally Robust Multi-Task Networks

We explore the utility of multi-task learning in improving performance of group-robustness approaches.

### Running Experiments 

Currently we only support the Celeb-A dataset and the following methodologies: ERM, JTT, MTL ERM, MTL JTT. Follow the below to run experiments for these. 


#### ERM:

Run

```
python train_erm.py
```

where hyperparameters should be specified in the following config [file](https://github.com/vliu15/mml-robustness/blob/main/configs/exp/erm.yaml). Make sure that groupings is a list of length 1 to not accidentally run a MTL experiment.

#### JTT:

Run

```
python train_jtt.py
```

where hyperparameters should be specified in the following config [file](https://github.com/vliu15/mml-robustness/blob/main/configs/exp/jtt.yaml) and correspondingingly in `./jtt_stage_1.yaml` and `./jtt_stage_2.yaml`. Make sure that groupings is a list of length 1 to not accidentally run a MTL experiment. 

#### MTL ERM:

Run

```
python train_erm.py
```

where hyperparameters should be specified in the following config [file](https://github.com/vliu15/mml-robustness/blob/main/configs/exp/erm.yaml). Make sure that groupings is a list of length greather than 1 to not accidentally run a STL experiment. 

#### MTL JTT:

Run

```
python train_jtt.py
```

where hyperparameters should be specified in the following config [file](https://github.com/vliu15/mml-robustness/blob/main/configs/exp/jtt.yaml) and correspondingingly in `./jtt_stage_1.yaml` and `./jtt_stage_2.yaml`. Make sure that groupings is a list of length greather than 1 to not accidentally run a STL experiment. 


#### Spurrious Correlate Identification:

To identifty spurious correlates of a given task label run:

```
./scripts/run_spurious_identification.sh
```
