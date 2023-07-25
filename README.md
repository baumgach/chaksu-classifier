## About this repo

This repo contains example code to train a ResNet18 classifier on the [MIMeta](https://www.l2l-challenge.org/data.html) version of the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629) breast mass classification task. The classifier is implemented in `pytorch-lightning`. Furthermore, a script is provided for training on the Tübingen ML Cloud infrastructure. 

## Setup 

Make a conda environment and install the required packages using `pip`. Unfortunately, I didn't have time yet to create a proper `requirements.txt`, so you have to manually check the imports. PR welcome! 

Copy the `mammo_mass` folder of the MIMeta dataset into the `./data` folder. 

If you want to run the code on the Tübingen ML Cloud, also create an empty `slurm_logs` folder. 

## Training 

You can train the classifier with default settings using 

````
python train.py --experiment_name="cbim" --use_data_augmentation
````

Using data augmentation provides the best results. Max epochs is set to 1000. The best validation AUC is automatically selected and is typically achieved around 200 epochs. 

If you want to run the code on the Tübingen ML Cloud, use the following command 

````
sbatch --partition=gpu-2080ti deploy.sh
````

## Monitoring the training using tensorboard

Start a tensorboard instance in the `runs` directory, and open tensorboard in your browser. 

````
tensorboard --logdir='./runs'
````

If you are using the ML Cloud, start a tensorboard in a tmux shell using a specific port, e.g. 

````
tensorboard --logdir=runs --port=2326
````

and then SSH onto the login node using port forwarding, i.e. 
````
ssh -L 2326:localhost:2326 slurm
````

This will make tensorboard available on your local browser on `localhost:2326`. 

## Testing 

Once checkpoints are written you can start testing the model using this command 

````
python test.py --checkpoints_dir=runs/cbim-with-aug-LR0.0001-WD0.0/version_0/checkpoints --checkpoint_identifier='auc'
````

where `--checkpoints_dir` points to the actual experiment name and `--checkpoint_identifier` allows you to choose between the model with the best validation `auc` or the lowest validation `loss`. If the argument is omitted, the latest model is used by default. Selecting by AUC provides better results. 

## Results 

The default model with data augmentation currently achieves an accuracy of 0.714 and an AUROC of 0.784 on the official test split. Note that the model achieves values >0.95 for both on the validation split. The reason for this is not quite clear yet, but it may be that the official test data is more challenging than the training data. 