import os 
import shutil 

class residual_save_path:
    model_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_residual'
    img_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/figures/four_peak_residual'
    data_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_residual'

class rar_save_path:
    model_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_rar'
    img_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/figures/four_peak_rar'
    data_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_rar'

class grad_save_path:
    model_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_grad'
    img_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/figures/four_peak_grad'
    data_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_grad'

class rar_grad_save_path:
    model_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_rar_grad'
    img_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/figures/four_peak_rar_grad'
    data_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_rar_grad'

class raw_save_path:
    model_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_raw'
    img_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/figures/four_peak_raw'
    data_save_path = '/home/gaozhiwei/python/adaptive_restart_pinn/data/four_peak_raw'

def create_path(model = residual_save_path):
    length = len(model.__dict__.keys()) - 4
    index = list(model.__dict__.values())
    for i in range(1, 1 + length):
        if not os.path.exists(index[i]):
            os.makedirs(index[i])

def delete_img_path(models):
    if os.path.exists(models.img_save_path):
        shutil.rmtree(models.img_save_path)
        os.makedirs(models.img_save_path)