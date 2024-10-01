import torch
import datetime
import os
import yaml
def save_model(path_models, model):
    """
    Overwrite previously saved model with new parameters.
    :param path_models: model directory
    :param model: instance of the model class to be saved
    """

    # os.system("rm -rf " + path_models + model.__class__.__name__ + ".pt")
    model_name = model.__class__.__name__ + ".pt"
    torch.save(model.state_dict(), path_models+'/'+model_name)
    print('model save at:',path_models+'/'+model_name)

def creat_model_save_path(path_models):
    now =datetime.datetime.now()
    data = now.strftime('%Y-%m-%d')
    time = now.strftime('%H-%M-%S')

    folder_name = path_models+'/'+data+'_'+time
    os.makedirs(folder_name,exist_ok=True)

    log_file=os.path.join(folder_name,'logs')
    os.makedirs(log_file, exist_ok=True)

    cog_files = os.path.join(folder_name,'configs.yaml')
    # os.makedirs(cog_files, exist_ok=True)
    return folder_name,log_file,cog_files

def save_configs(configs,save_path):
    with open(save_path,'w') as f:
        yaml.dump(configs,f)
    print("configs save at",save_path)

