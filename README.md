## About this repo

This repo contains example code to train a ResNet18 or ResNet50 classifier on the Chaksu dataset. The classifier is implemented in `pytorch-lightning`. Furthermore, a script is provided for training on the Tübingen ML Cloud infrastructure. 

## Training 

You can train the classifier with default settings using 

````
python train.py --experiment_name="chaksu" 
````

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
python test.py --checkpoints_dir=<path-to-your-checkpoint-folder> --checkpoint_identifier='auc'
````

where `--checkpoints_dir` points to the actual experiment name and `--checkpoint_identifier` allows you to choose between the model with the best validation `auc` or the lowest validation `loss`. If the argument is omitted, the latest model is used by default. Selecting by AUC provides better results. 

