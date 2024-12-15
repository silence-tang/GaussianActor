from lgm_based.data_module_v3 import ZJUMOCAPDataModule
import numpy as np

data_module = ZJUMOCAPDataModule(
    data_dir='./data/zju_mocap',
    subject='CoreView_' + '315',
    dataset_folder_aux_undistorted='./data/zju_mocap_pre2',
    image_size=256,
    view_image_size=512,
    num_workers=1,
    test_mode = 'np',
    erode_mask = True,
)

data_module.setup(stage='test')

dataset = data_module.test_dataset
a = dataset[0]

image_list = []
rays_o_list = []
rays_d_list = []

for ind, view in enumerate(a['current_data_list']):
    image = view['image']
    mask = view['image_mask']
    rays_o = view['cam_loc']
    rays_d = view['sample_ray_dirs_all']

    image_wbg = image.copy()
    image_wbg[mask == 0] = 1

    image_list.append(image_wbg)
    rays_o_list.append(rays_o)
    rays_d_list.append(rays_d)

images = np.stack(image_list, axis=0)
rays_o = np.stack(rays_o_list, axis=0)
rays_d = np.stack(rays_d_list, axis=0)

with open('tmp/lgm/test.npy', 'wb') as f:
    np.save(f, images)
    np.save(f, rays_o)
    np.save(f, rays_d)
...
