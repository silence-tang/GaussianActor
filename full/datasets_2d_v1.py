import glob
import json
import numbers
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import smplx

# utils from ARAH
'''https://github.com/taconite/arah-release.git'''

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    # near = near[mask_at_box] / norm_d[mask_at_box, 0]
    # far = far[mask_at_box] / norm_d[mask_at_box, 0]
    near = near / norm_d[..., 0]
    far = far / norm_d[..., 0]
    return near, far, mask_at_box

# data

class ZJUMOCAPDataset(Dataset):
    def __init__(self,
        dataset_folder,
        dataset_folder_aux_undistorted=None,
        subjects=['CoreView_313'],
        mode='train',
        img_size=(512, 512),
        sampling_rate=1,
        start_frame=0,
        end_frame=-1,
        views=[],
        sample_views=-1,  # 12 for regular training and 17 for train controlnet
        box_margin=0.05,
        erode_mask=True,
        repeat=1):
        ''' Initialization of the the ZJU-MoCap dataset.

        !!! FRAME VIEW Ver. !!!

        Args:
            dataset_folder (str): dataset folder
            subjects (list of strs): which subjects to use
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            img_size (int or tuple of ints): target image size
            sampling_rate (int): sampling rate for video frames
            start_frame (int): start frame of the video
            end_frame (int): end frame of the video
            views (list of strs): which views to use
            box_margin (float): bounding box margin added to SMPL bounding box. This bounding box is used to determine sampling region in an image
            erode_mask (bool): whether to erode ground-truth foreground masks, such that boundary pixels of masks are ignored
            repeat (int): len * N, id // N
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.box_margin = box_margin
        self.erode_mask = erode_mask
        self.repeat = repeat

        assert sample_views <= len(views)

        # pass
        if sample_views == -1:
            sample_views = len(views)

        self.sample_views = sample_views
        
        self.random_select_views = self.sample_views < len(views)

        self.faces = np.load('/home/xxx/GaussianActor/body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('/home/xxx/GaussianActor/body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('/home/xxx/GaussianActor/body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('/home/xxx/GaussianActor/body_models/misc/J_regressors.npz'))

        # self.img_size=512
        if isinstance(img_size, numbers.Number):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = img_size

        self.rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()  # pos 45
        self.rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix() # neg 45
        # self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
        #     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
        # self.ktree_children = np.array([-1,  4,  5,  6,  7,  8,  9,  10,  11,  -1,  -1,  -1,
        #     15,  16,  17, -1, 18, 19, 20, 21, 22, 23, -1, -1], dtype=np.int32)

        assert (len(subjects) == 1) # TODO: we only support per-subject training at this point

        with open(os.path.join(dataset_folder, subjects[0], 'cam_params.json'), 'r') as f:
            cameras = json.load(f)

        self.cameras = cameras

        # pass
        if len(views) == 0:
            cam_names = cameras['all_cam_names']
        else:
            cam_names = views

        # ['1', '2', '3', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '17', '18', '19', '23']
        self.cam_names = cam_names

        self.homo_2d = self.init_grid_homo_2d(img_size[0], img_size[1])

        # Get all data
        self.data = []
        self.data_by_frame = {}
        # self.data_by_frame_test_cam = {}

        for subject in subjects:

            subject_dir = os.path.join(dataset_folder, subject)

            if end_frame > 0:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame:end_frame:sampling_rate]
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame::sampling_rate]

            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
                frames = np.arange(len(img_files)).tolist()

                if end_frame > 0:
                    img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[start_frame:end_frame:sampling_rate]
                    mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[start_frame:end_frame:sampling_rate]
                    frames = frames[start_frame:end_frame:sampling_rate]
                else:
                    img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[start_frame::sampling_rate]
                    mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[start_frame::sampling_rate]
                    frames = frames[start_frame::sampling_rate]

                assert (len(model_files) == len(img_files) and len(mask_files) == len(img_files))

                for d_idx, (f_idx, img_file, mask_file, model_file) in enumerate(zip(frames, img_files, mask_files, model_files)):
                    data_item = {
                        'subject': subject,
                        'gender': 'neutral',
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'frame_idx': f_idx,
                        'data_idx': d_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                        'aux_img_file': os.path.join(dataset_folder_aux_undistorted, *img_file.replace("\\", "/").split('/')[-3:]) if dataset_folder_aux_undistorted is not None else None,
                        'aux_mask_file': os.path.join(dataset_folder_aux_undistorted, *mask_file.replace("\\", "/").split('/')[-3:]) if dataset_folder_aux_undistorted is not None else None,
                    }
                    self.data.append(data_item)
                    frame_data_list = self.data_by_frame.get(f_idx)
                    if frame_data_list is not None:
                        frame_data_list.append(data_item)
                    else:
                        self.data_by_frame[f_idx] = [data_item]

        self.data_by_frame_list = list(self.data_by_frame.keys())
    
    # def unnormalize_canonical_points(self, pts, coord_min, coord_max, center):
    #     padding = (coord_max - coord_min) * 0.05
    #     pts = (pts / 2.0 + 0.5) * 1.1 * (coord_max - coord_min) + coord_min - padding +  center
    #     return pts

    # def normalize_canonical_points(self, pts, coord_min, coord_max, center):
    #     pts -= center
    #     padding = (coord_max - coord_min) * 0.05
    #     pts = (pts - coord_min + padding) / (coord_max - coord_min) / 1.1
    #     pts -= 0.5
    #     pts *= 2.
    #     return pts

    def get_meshgrid(self, height, width):
        Y, X = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )
        grid_map = np.stack([X, Y], axis=-1)  # (height, width, 2)
        return grid_map

    def get_homo_2d_from_xy(self, xy):
        H, W = xy.shape[0], xy.shape[1]
        homo_ones = np.ones((H, W, 1), dtype=np.float32)
        homo_2d = np.concatenate((xy, homo_ones), axis=2)
        return homo_2d

    def get_homo_2d(self, height, width):
        xy = self.get_meshgrid(height, width)
        homo_2d = self.get_homo_2d_from_xy(xy)
        return homo_2d

    def init_grid_homo_2d(self, height, width):
        homo_2d = self.get_homo_2d(height, width)
        homo_2d = homo_2d    # (height*width, 3)
        return homo_2d

    def normalize_vectors(self, x):
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        eps = 1e-12
        x = x / (norm + eps)
        return x

    def get_camera_location(self, R, t):
        cam_loc = np.dot(-R.T, t)
        return cam_loc

    def get_camera_rays(self, R, homo_2d):
        rays = np.dot(homo_2d, R) # (H*W, 3)
        rays = self.normalize_vectors(rays) # (H*W, 3)
        return rays

    def get_mask(self, mask_in):
        mask = (mask_in != 0).astype(np.uint8)

        if self.erode_mask or self.mode in ['val', 'test']:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            mask_erode = cv2.erode(mask.copy(), kernel)
            mask_dilate = cv2.dilate(mask.copy(), kernel)
            mask[(mask_dilate - mask_erode) == 1] = 100

        return mask
    
    def get_02v_bone_transforms(self, Jtr, rot45p, rot45n):
        # Specify the bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

        # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
        chain = [1, 4, 7, 10]
        rot = rot45p.copy()
        for i, j_idx in enumerate(chain):
            bone_transforms_02v[j_idx, :3, :3] = rot
            t = Jtr[j_idx].copy()
            if i > 0:
                parent = chain[i-1]
                t_p = Jtr[parent].copy()
                t = np.dot(rot, t - t_p)
                t += bone_transforms_02v[parent, :3, -1].copy()

            bone_transforms_02v[j_idx, :3, -1] = t

        bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
        # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
        chain = [2, 5, 8, 11]
        rot = rot45n.copy()
        for i, j_idx in enumerate(chain):
            bone_transforms_02v[j_idx, :3, :3] = rot
            t = Jtr[j_idx].copy()
            if i > 0:
                parent = chain[i-1]
                t_p = Jtr[parent].copy()
                t = np.dot(rot, t - t_p)
                t += bone_transforms_02v[parent, :3, -1].copy()

            bone_transforms_02v[j_idx, :3, -1] = t

        bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

        return bone_transforms_02v


    def __len__(self):
        ''' Returns the length of the dataset.'''
        return len(self.data_by_frame_list) * self.repeat
    

    def get_single_item(
            self,
            image,
            mask,
            mask_erode,
            K,
            R,
            cam_trans, # T
            cam_loc,
            cam_idx,
            frame_idx,
            data_idx,
            min_xyz=None,
            max_xyz=None,
    ):
        
        img_size = self.img_size
        homo_2d = self.homo_2d
        orig_img_size = (image.shape[0], image.shape[1])

        # Resize image
        img_crop = cv2.resize(image, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        mask_erode_crop = cv2.resize(mask_erode, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        img_crop = img_crop.astype(np.float32)
        img_crop /= 255.0
        side = max(orig_img_size)

        # Update camera parameters
        principal_point = K[:2, -1].reshape(-1).astype(np.float32)
        focal_length = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
        focal_length = focal_length / side  * max(img_size)
        principal_point = principal_point / side * max(img_size)

        K = K.copy()
        K[:2, -1] = principal_point
        K[0, 0] = focal_length[0]
        K[1, 1] = focal_length[1]

        K_inv = np.linalg.inv(K)    # for mapping rays from camera space to world space

        data = {
            'center_cam': principal_point,
            'focal_length': focal_length,
            'K': K,
            'R': R,
            'T': cam_trans,
            'cam_loc': cam_loc,
            'image': img_crop,
            'image_mask': mask_crop,
            'image_mask_erode': mask_erode_crop,
            'img_height': int(img_size[0]),
            'img_width': int(img_size[1]),
            'cam_idx': int(cam_idx),
            'frame_idx': int(frame_idx),
            'data_idx': int(data_idx),
        }

        # Get foreground mask bounding box from which to sample rays
        min_xyz = min_xyz.copy()
        max_xyz = max_xyz.copy()
        min_xyz -= self.box_margin
        max_xyz += self.box_margin
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), img_size[0], img_size[1])
        y_inds, x_inds = np.where(bound_mask != 0)

        sampled_pixels = img_crop[y_inds, x_inds, :].copy()
        bg_sample_mask = mask_erode_crop == 0
        sampled_bg_mask = bg_sample_mask[y_inds, x_inds].copy()
        sampled_pixels[sampled_bg_mask] = 0
        sampled_uv = np.dot(homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)
        sampled_rays = self.get_camera_rays(R, sampled_uv)

        near, far, mask_at_box = get_near_far(bounds, np.broadcast_to(cam_loc, sampled_rays.shape), sampled_rays)

        image_mask = np.zeros(mask_crop.shape, dtype=bool)
        image_mask[y_inds[mask_at_box], x_inds[mask_at_box]] = True

        data.update({
            'sample_image_mask': image_mask,
        })

        return data
    
    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of frame data point
        '''

        idx = idx // self.repeat

        data_list = self.data_by_frame[self.data_by_frame_list[idx]]

        data_path = data_list[0]['model_file']
        frame_idx = data_list[0]['frame_idx']
        gender = data_list[0]['gender']
        data = {}

        model_dict = np.load(data_path)
        
        # 3D models and points
        trans = model_dict['trans'].astype(np.float32)
        minimal_shape = model_dict['minimal_shape']

        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        n_smpl_points = minimal_shape.shape[0]
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        # pose_quat = pose_quat.reshape(-1)
        pose_mat_full = pose.as_matrix()                 # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()         # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape([-1, 9])   # 24 x 9, root rotation is set to identity

        pose_rot_full = pose_mat_full.reshape([-1, 9])   # 24 x 9, including root rotation

        # Minimally clothed shape with pose-blend shapes
        posedir = self.posedirs[gender]
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        ident = np.eye(3)
        pose_feature = (pose_mat - ident).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape += pose_offsets

        # Get posed minimally-clothed shape
        skinning_weights = self.skinning_weights[gender]
        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        minimal_body_vertices = (np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans).astype(np.float32)

        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = self.get_02v_bone_transforms(Jtr, self.rot45p, self.rot45n)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        minimal_shape_v = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        # Normalize conanical pose points with GT full-body scales.
        center = np.mean(minimal_shape_v, axis=0)
        minimal_shape_v_centered = minimal_shape_v - center
        coord_max = minimal_shape_v_centered.max()
        coord_min = minimal_shape_v_centered.min()

        padding = (coord_max - coord_min) * 0.05

        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - coord_min + padding) / (coord_max - coord_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # Get centroid of each part
        Jtr_mid = np.zeros([24, 3], dtype=np.float32)
        part_idx = skinning_weights.argmax(-1)
        for j_idx in range(24):
            Jtr_mid[j_idx, :] = np.mean(minimal_body_vertices[part_idx == j_idx, :], axis=0)

        min_xyz = np.min(minimal_body_vertices, axis=0)
        max_xyz = np.max(minimal_body_vertices, axis=0)

        selected_data = np.random.choice(data_list, self.sample_views, replace=False) if self.random_select_views else data_list

        current_data_list = []

        for item in selected_data:
            img_path = item['img_file']
            mask_path = item['mask_file']
            cam_name = item['cam_name']
            cam_idx = item['cam_idx']
            frame_idx = item['frame_idx']
            data_idx = item['data_idx']

            aux_img_path = item['aux_img_file']
            aux_mask_path = item['aux_mask_file']

            if aux_img_path is not None:
                img_path = aux_img_path
            if aux_mask_path is not None:
                mask_path = aux_mask_path

            # Load and undistort image and mask
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_erode = self.get_mask(mask)

            K = np.array(self.cameras[cam_name]['K'], dtype=np.float32)
            dist = np.array(self.cameras[cam_name]['D'], dtype=np.float32).ravel()
            R = np.array(self.cameras[cam_name]['R'], np.float32)
            cam_trans = np.array(self.cameras[cam_name]['T'], np.float32).ravel()

            cam_loc = self.get_camera_location(R, cam_trans)

            if aux_img_path is None:
                image = cv2.undistort(image, K, dist, None)
            if aux_mask_path is None:
                mask = cv2.undistort(mask, K, dist, None)
                mask_erode = cv2.undistort(mask_erode, K, dist, None)

            current_item_data = self.get_single_item(
                image,
                mask,
                mask_erode,
                K,
                R,
                cam_trans,
                cam_loc,
                cam_idx,
                frame_idx,
                data_idx,
                min_xyz=min_xyz,
                max_xyz=max_xyz,
            )

            current_item_data.update({
                'cam_name': cam_name,
            })

            current_data_list.append(current_item_data)

        T_02p = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4]) + np.pad(trans[:, None], ((0, 1), (3, 0)))  # [6890, 4, 4]
        T_p20 = np.linalg.inv(T_02p)     # [6890, 4, 4]
        T_02v = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])  # [6890, 4, 4]
        # T_v20 = np.linalg.inv(T_02v)   # [6890, 4, 4]
        T_p2v = np.matmul(T_02v, T_p20)  # [6890, 4, 4]

        data = {
            'trans': trans,
            'bone_transforms': bone_transforms.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v.astype(np.float32),
            'coord_max': coord_max.astype(np.float32),
            'coord_min': coord_min.astype(np.float32),
            'center': center.astype(np.float32),
            'minimal_shape': minimal_shape_v.astype(np.float32),
            'smpl_vertices': minimal_body_vertices.astype(np.float32),
            'T_inv': T_p2v.astype(np.float32),
            'skinning_weights': skinning_weights.astype(np.float32),
            'root_orient': root_orient,
            'pose_hand': pose_hand,
            'pose_body': pose_body,
            'Jtr_mid': Jtr_mid,
            'rots': pose_rot.astype(np.float32),
            'Jtrs': Jtr_norm.astype(np.float32),
            'rots_full': pose_rot_full.astype(np.float32),
            'Jtrs_posed': Jtr_posed.astype(np.float32),
            'gender': gender,
            'idx': int(idx),
            'current_data_list': current_data_list,
        }

        return data
