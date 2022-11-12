import json
import numpy as np
import torch

with open(r'./common/camera.json') as f:
    camera_parameter = json.load(f)


def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
      P: Nx3 points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: (scalar) Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
    Returns
      Proj: Nx2 points in pixel space
      D: 1xN depth of each point in camera space
      radial: 1xN radial distortion per point
      tan: 1xN tangential distortion per point
      r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = torch.mm(R, (P.T - T))  # rotate and translate for tensor
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2
    radial = 1 + torch.einsum('ij,ij->j', k.repeat((1, N)), torch.stack([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    XXX = XX * (radial + tan).repeat([2, 1]) + torch.outer(torch.tensor([p[1], p[0]]).reshape(-1), r2)
    Proj = (f * XXX) + c
    Proj = Proj.T
    D = X[2,]
    return Proj, D, radial, tan, r2


def zero_the_root(pose, root_idx):
    if isinstance(pose, np.ndarray):
        # center at root
        root_pose = []
        for i in range(pose.shape[0]):
            root_pose.append(pose[i, root_idx, :])
            pose[i, :, :] = pose[i, :, :] - pose[i, root_idx, :]
            # remove root
        pose = np.delete(pose, root_idx, 1)  # axis [n, j, x/y]
        root_pose = np.asarray(root_pose)

        return pose, root_pose

    elif torch.is_tensor(pose):
        pose1 = pose.clone()
        root_pose = pose1[:, root_idx, :].reshape(pose.shape[0], -1, pose.shape[2])

        for i in range(pose.shape[0]):
            ## or the loss of root node information, create a root node
            if sum(pose1[i, root_idx, :]) == 0:
                pose1[i, root_idx, :] = (pose1[i, 1, :] - pose1[i, 4, :]) / 2

            pose1[i, :, :] = pose1[i, :, :] - pose1[i, root_idx, :]
        # pose1 = pose1[:, 1:, :]
        return pose1  # , root_pose

    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def generate_proj_matrix(subject):
    proj_matrix = []
    for cam_idx in range(4):
        R, T, f, c, k, p, name = camera_parameter[str((subject, cam_idx + 1))]
        R, T = np.array(R), np.array(T)
        K = np.array([[f[0][0], 0., c[0][0]], [0., f[1][0], c[1][0]], [0., 0., 1.]]).astype(np.double)
        T = np.dot(R, np.negative(T))
        if len(T.shape) < 2:
            T = np.expand_dims(T, axis=-1)
        temp_proj_matrix = np.dot(K, np.concatenate((R, T), axis=1))
        proj_matrix.append(temp_proj_matrix)
    return proj_matrix


def reprojection_loss(gt_3D, target_2D, subject):
    gt_3D, input_2D = gt_3D.cpu(), target_2D.clone().cpu()

    batch_size = gt_3D.shape[0]
    multi = 4
    project_points_batch = []

    for cam_idx in range(multi):
        proj_points = []
        for batch_idx in range(batch_size):
            subject_temp = subject[batch_idx]
            subject_temp = int(subject_temp[1:])
            R, T, f, c, k, p, name = camera_parameter[str((subject_temp, cam_idx + 1))]
            R, T, f, c, k, p = torch.tensor(R), torch.tensor(T), torch.tensor(f), torch.tensor(c), torch.tensor(
                k), torch.tensor(p)
            proj_point, _, _, _, _ = project_point_radial(gt_3D[batch_idx] * 1000, R, T, f, c, k, p)

            w = 1000
            if cam_idx == 0 or 3:
                h = 1002
            else:
                h = 1000
            proj_point = proj_point / w * 2 - torch.tensor([1, h / w])
            proj_points.append(proj_point)

        proj_points = torch.stack(proj_points)
        proj_points = zero_the_root(proj_points, 0)
        input_2D[:, cam_idx, :, :] = zero_the_root(input_2D[:, cam_idx, :, :], 0)

        project_points_batch.append(proj_points)
    project_points_batch = torch.stack(project_points_batch).transpose(1, 0)

    loss_batch = torch.mean(torch.norm(project_points_batch - input_2D, p=1, dim=len(input_2D.shape) - 1),
                            dim=len(input_2D.shape) - 2)
    return torch.mean(loss_batch, dim=1)
