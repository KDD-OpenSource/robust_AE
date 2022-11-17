# On Training and Verifying Robust Autoencoders
This repository contains the code and experimental configuration files for the DSAA2022 Paper On Training and Verifying Robust Autoencoders

## Usage
Please note that in order to make use of our solution framework you need to install [Marabou](https://github.com/NeuralNetworkVerification/Marabou)

```bash
git clone git://github.com/KDD-OpenSource/robust_AE.git  
virtualenv venv -p /usr/bin/python3  
source venv/bin/activate  
pip install -r requirements.txt  
```

## Reproduction of Experiments:
To reproduce the results of the any experiment in the paper first run 
```bash
python3 main.py configs/reprod/dsaa/config_train_EXPERIMENT.cfg
```
for your choice of experiment. This creates the necessary models.
Thereafter store the model in e.g. 'models/reprod/autoencoder_EXPERIMENT'
and run
```bash
python3 main.py configs/reprod/dsaa/config_test_EXPERIMENT.cfg
```
Make sure that the path in the respective test configuration file poins to where you have stored the previously trained models.
You will find the results in the folder 'reports'.

## Authors/Contributors
* [Benedikt Böing](https://github.com/bboeing)
* [Emmanuel Müller](https://github.com/emmanuel-mueller)
