import json
import torch
import numpy as np

with open(r'./common/camera.json') as f:
    camera_parameter = json.load(f)


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
            pose1[i, :, :] = pose1[i, :, :] - pose1[i, root_idx, :]
        # pose1 = pose1[:, 1:, :]
        return pose1  # , root_pose

    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_loss(output_3D, input_2D_points, subject):
    input_2D = input_2D_points.clone()
    input_2D, output_3D = input_2D.cpu(), output_3D.cpu()
    batch_size = output_3D.shape[0]
    multi = 4
    triangulate_3d_points_batch = []
    for batch_idx in range(batch_size):
        ## input_2D denormalization
        input_2D[batch_idx, 0, :, :] = (input_2D[batch_idx, 0, :, :] + torch.tensor([1, 1.002])) / 2 * 1000
        input_2D[batch_idx, 3, :, :] = (input_2D[batch_idx, 3, :, :] + torch.tensor([1, 1.002])) / 2 * 1000
        input_2D[batch_idx, 1, :, :] = (input_2D[batch_idx, 1, :, :] + torch.tensor([1, 1])) / 2 * 1000
        input_2D[batch_idx, 2, :, :] = (input_2D[batch_idx, 2, :, :] + torch.tensor([1, 1])) / 2 * 1000

        ## get the projection matrix
        subject_temp = subject[batch_idx]
        subject_temp = int(subject_temp[1:])
        proj_matrixs = generate_proj_matrix(subject_temp)
        proj_matrixs = torch.from_numpy(np.array(proj_matrixs))

        triangulate_3d_points = []
        for joints_idx in range(17):
            A = proj_matrixs[:, 2:3].expand(multi, 2, 4) * input_2D[batch_idx, :, joints_idx, :].view(multi, 2, 1)
            A -= proj_matrixs[:, :2]
            u, s, vh = torch.svd(A.view(-1, 4))
            point_3d_homo = -vh[:, 3]
            triangulate_3d = (point_3d_homo.T[:-1] / point_3d_homo.T[-1]).T
            triangulate_3d_points.append(triangulate_3d / 1000)

        triangulate_3d_points = torch.stack(triangulate_3d_points, dim=0)

        ## For the loss of root node information, create a root node
        if sum(triangulate_3d_points[0, :]) == 0:
            triangulate_3d_points[0, :] = (triangulate_3d_points[1, :] + triangulate_3d_points[4, :]) / 2

        triangulate_3d_points_batch.append(triangulate_3d_points)
    triangulate_3d_points_batch = torch.stack(triangulate_3d_points_batch, dim=0)
    triangulate_3d_points_batch = zero_the_root(triangulate_3d_points_batch, 0)

    loss_batch = torch.mean(torch.norm(triangulate_3d_points_batch - output_3D, p=1, dim=len(output_3D.shape) - 1),
                            dim=len(output_3D.shape) - 2)
    return loss_batch
