#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp, pixels_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    #
    ema_depth_normal_01_loss_for_log = 0.0
    ema_distance_01_loss_for_log = 0.0
    # ema_pd_01_loss_for_log = 0.0

    normal_loss, geo_loss, ncc_loss, depth_normal_01_loss, distance_01_loss, pd_01_loss = None, None, None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter = first_iter + 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss = loss + opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss = loss + (normal_loss)

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                # geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=True)
                #
                # import pdb; pdb.set_trace()
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    # geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    # loss = loss + geo_loss



                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss = loss + ncc_loss





                        # jiabo
                        # import pdb; pdb.set_trace()
                        # compute multi_view depth_normal_01_loss
                        depth0_normal0 = render_pkg["depth_normal"]
                        pixels_0 = torch.stack([ix, iy], dim=-1).float().to(depth0_normal0.device)
                        pixels_0_valid = pixels_0.reshape(-1,2)[valid_indices]  # 根据有效索引重新排列像素坐标
                        pixels_1_valid = pixels_warp(H_ref_to_neareast.reshape(-1,3,3), pixels_0_valid)  # 使用单应性矩阵对原始像素进行变换  
                        pixels_0_valid[:, 0] = 2 * pixels_0_valid[:, 0] / (W - 1) - 1.0  # 将x坐标归一化到[-1,1]  
                        pixels_0_valid[:, 1] = 2 * pixels_0_valid[:, 1] / (H - 1) - 1.0  # 将y坐标归一化到[-1,1]  
                        depth0_normal0_valid = gaussians.get_pixels_normal_in_depth_normal(viewpoint_cam, depth0_normal0, pixels_0_valid) 
                        R0 = torch.tensor(viewpoint_cam.R).float().cuda()  # 获取最近相机的旋转矩阵并移到GPU  
                        # T0 = torch.tensor(viewpoint_cam.T).float().cuda()  # 获取最近相机的平移向量并移到GPU  
                        depth0_normal0_valid_T = depth0_normal0_valid.T  # 形状变为 (102400, 3)
                        dn0_valid = depth0_normal0_valid_T@R0.transpose(-1,-2) 
                        dn_valid_0 = dn0_valid.T

                        depth1_normal1 = nearest_render_pkg["depth_normal"]
                        pixels_1_valid[:, 0] = 2 * pixels_1_valid[:, 0] / (W - 1) - 1.0  # 将x坐标归一化到[-1,1]  
                        pixels_1_valid[:, 1] = 2 * pixels_1_valid[:, 1] / (H - 1) - 1.0  # 将y坐标归一化到[-1,1]  
                        depth1_normal1_valid = gaussians.get_pixels_normal_in_depth_normal(nearest_cam, depth1_normal1, pixels_1_valid)  # 在最近的相机的深度法线图中找到对应的法线并生成掩码  
                        R1 = torch.tensor(nearest_cam.R).float().cuda()  # 获取最近相机的旋转矩阵并移到GPU  
                        # T1 = torch.tensor(nearest_cam.T).float().cuda()  # 获取最近相机的平移向量并移到GPU  
                        depth1_normal1_valid_T = depth1_normal1_valid.T  # 形状变为 (102400, 3)
                        dn1_valid = depth1_normal1_valid_T@R1.transpose(-1,-2) 
                        dn_valid_1 = dn1_valid.T
                        # 计算法线余弦角？
                        # import pdb; pdb.set_trace()

                        # diff_sum = (dn_valid_0 - dn_valid_1).abs().sum(0)
                        # # 对张量进行排序
                        # sorted_diff_sum, _ = torch.sort(diff_sum)
                        # # 计算中位数
                        # n = diff_sum.numel()  # 获取张量中的元素数量
                        # median = None
                        # if n % 2 == 1:
                        #     # 如果元素数量是奇数，中位数是中间的那个元素
                        #     median = sorted_diff_sum[n // 2]
                        # else:
                        #     # 如果元素数量是偶数，中位数是中间两个元素的平均值
                        #     median = (sorted_diff_sum[n // 2 - 1] + sorted_diff_sum[n // 2]) / 2
                        # print(median)
                        # scan122 0.1247 1.2747 1.4179 1.4815 1.0114 1.4903 0.1129 1.4181 0.2694
                        # scan65 1.4134 1.3890 1.3712 1.3471 1.0806 1.0796 1.1263 0.5347 0.1910 1.3610 1.3407



                        # dn_mask = (depth0_normal0_valid - depth1_normal1_valid).abs().sum(0) < dn_noise_th
                        dn_mask = (dn_valid_0 - dn_valid_1).abs().sum(0) < 0.52
                        # scan122
                        # 0.8 0.33676942342012983 1 0.3366944878008501 1.2 0.33517956663682136 1.5 0.3328736850139251 
                        # 1.7 0.3328835666971438 1.9 0.33058022223996963 2 0.33240557373060065 2.3 0.33292113119138333

                        # scan65 
                        # 0.8 0.6058430273824661 1 0.6060768905622127 1.2 0.6501554445015649 1.5 0.6336188789138292 
                        # 1.8 0.6354486129005621



                        weight = opt.single_view_weight
                        image_weight0 = (1.0 - get_img_grad_weight(gt_image))
                        # import pdb; pdb.set_trace()
                        image_weight1 = image_weight0.reshape(-1)[valid_indices]
                        image_weight2 = image_weight1 * dn_mask
                        image_weight2[~dn_mask] = 0
                        image_weight = (image_weight2).clamp(0,1).detach() ** 2
                        if not opt.wo_image_weight:
                            # image_weight = erode(image_weight[None,None]).squeeze()
                            depth_normal_01_loss = weight * (image_weight * ((dn_valid_0 - dn_valid_1).abs().sum(0))).mean()
                        else:
                            depth_normal_01_loss = weight * (((dn_valid_0 - dn_valid_1)).abs().sum(0)).mean()
                        loss = loss + depth_normal_01_loss


                            
                        # jiabo
                        rendered_pd0 = render_pkg["plane_depth"]
                        uv_0 = torch.stack([ix, iy], dim=-1).float().to(rendered_pd0.device)
                        uv_0_valid = uv_0.reshape(-1,2)[valid_indices]  # 根据有效索引重新排列像素坐标
                        uv_1_valid = pixels_warp(H_ref_to_neareast.reshape(-1,3,3), uv_0_valid)  # 使用单应性矩阵对原始像素进行变换  
                        uv_0_valid[:, 0] = 2 * uv_0_valid[:, 0] / (W - 1) - 1.0  # 将x坐标归一化到[-1,1]  
                        uv_0_valid[:, 1] = 2 * uv_0_valid[:, 1] / (H - 1) - 1.0  # 将y坐标归一化到[-1,1]  
                        rendered_pd0_valid = gaussians.get_uv_depth_in_plane_depth(viewpoint_cam, rendered_pd0, uv_0_valid) 
                        coord0_world_valid = gaussians.get_coord_from_depth(viewpoint_cam, rendered_pd0_valid, valid_indices)  # 从当前视图的深度图中反投影点到三维  
                        coord0_in_nearest_cam = coord0_world_valid @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]  # 将点转换到最近的相机坐标系  
                        
                        normal1 = nearest_render_pkg["rendered_normal"]
                        uv_1_valid[:, 0] = 2 * uv_1_valid[:, 0] / (W - 1) - 1.0  # 将x坐标归一化到[-1,1]  
                        uv_1_valid[:, 1] = 2 * uv_1_valid[:, 1] / (H - 1) - 1.0  # 将y坐标归一化到[-1,1]  
                        normal1_valid = gaussians.get_uv_normal_in_rendered_normal(viewpoint_cam, normal1, uv_1_valid) 
                        normal1_valid_transposed = normal1_valid.T
                        # import pdb; pdb.set_trace()
                        distance_0_to_1_valid = torch.sum(coord0_in_nearest_cam * normal1_valid_transposed, dim=1).abs().unsqueeze(0)

                        rendered_distance1 = nearest_render_pkg["rendered_distance"] 
                        rendered_distance1_valid = gaussians.get_uv_distance_in_rendered_distance(nearest_cam, rendered_distance1, uv_1_valid)  # 在最近的相机的深度图中找到对应的深度并生成掩码  
                        
                        # import pdb; pdb.set_trace()
                        # diff_sum = (distance_0_to_1_valid - rendered_distance1_valid).abs().sum(0)
                        # # 对张量进行排序
                        # sorted_diff_sum, _ = torch.sort(diff_sum)
                        # # 计算中位数
                        # n = diff_sum.numel()  # 获取张量中的元素数量
                        # median = None
                        # if n % 2 == 1:
                        #     # 如果元素数量是奇数，中位数是中间的那个元素
                        #     median = sorted_diff_sum[n // 2]
                        # else:
                        #     # 如果元素数量是偶数，中位数是中间两个元素的平均值
                        #     median = (sorted_diff_sum[n // 2 - 1] + sorted_diff_sum[n // 2]) / 2
                        # print(median)
                        # # scan65 dnmask 0.52 distance 0.05
                        # # 0.1147 0.3170 0.1734 0.3352 0.3743 0.1225 0.0626 0.0247 0.1219 0.2386 0.1509
                        # # scan83 dnmask 0.52 distance 0.05
                        # # 0.0200 0.1216 0.2101
                        
                        # 计算每个元素表示对应坐标点之间差异
                        distance_mask = (distance_0_to_1_valid - rendered_distance1_valid).abs().sum(0) < 0.03
                        # 0.15 0.33606325934667813 0.2 0.33423037979370873 0.5 0.33688411292122633 1 0.33605794517975635 3 0.33800320435597075 5 0.33578844747196746


                        # scan65 dnmask 1 (0.6133213576648497)
                        # +distance_mask 
                        # 0.1 0.6499996894730213 0.05

                        # 按三维点差异计算权重
                        distance_weights = (1.0 / torch.exp((distance_0_to_1_valid - rendered_distance1_valid).abs().sum(0))).detach()
                        distance_weights[~distance_mask] = 0
                        # import pdb; pdb.set_trace()
                        # distance_weights = image_weight1 * distance_mask
                        # distance_weights[~distance_mask] = 0
                        distance_weight = opt.multi_view_distance_weight
                        distance_01_loss = distance_weight * ((distance_weights * (distance_0_to_1_valid - rendered_distance1_valid).abs().sum(0))[distance_mask]).mean()
                        loss = loss + distance_01_loss 



        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            #
            ema_depth_normal_01_loss_for_log = 0.4 * depth_normal_01_loss.item() if depth_normal_01_loss is not None else 0.0 + 0.6 * ema_depth_normal_01_loss_for_log
            ema_distance_01_loss_for_log = 0.4 * distance_01_loss.item() if distance_01_loss is not None else 0.0 + 0.6 * ema_distance_01_loss_for_log
            # ema_pd_01_loss_for_log = 0.4 * pd_01_loss.item() if pd_01_loss is not None else 0.0 + 0.6 * ema_pd_01_loss_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "Single": f"{ema_single_view_for_log:.{4}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{4}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{4}f}",
                    #
                    "normal_01": f"{ema_depth_normal_01_loss_for_log:.{4}f}",
                    "distance_01": f"{ema_distance_01_loss_for_log:.{4}f}",
                    # "pd_01": f"{ema_pd_01_loss_for_log:.{4}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] = observe_cnt[out_observe > 0] + 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test = l1_test + l1_loss(image, gt_image).mean().double()
                    psnr_test = psnr_test + psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
