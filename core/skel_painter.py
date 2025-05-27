import cv2
import numpy as np
from matplotlib import pyplot as plt

from core.skel_def import skel_def_dict, skel_col_dict


def draw_skel_for_each_view(skel3d_set, proj_mat, views_path, type='BODY25', view_set=None):
    """

    :param skel3d_set: body x (batch) x joint x channel
    :param proj_mat:
    :param views_path:
    :param type:
    :param view_set:
    :return:
    """
    if len(skel3d_set.shape) == 4:
        skel3d_set = skel3d_set[:, 0, ::]
    limb_map = skel_def_dict[type]['limb_map']
    assert len(proj_mat) == len(views_path)
    if view_set is None:
        view_set = []
        for view_path_each in views_path:
            view_set.append(cv2.imread(view_path_each))
    assert len(view_set) == len(views_path)

    for skelId, skel3d in enumerate(skel3d_set):
        # make sure data is not empty
        if not len(skel3d):
            continue
        skel3d_data = np.hstack((skel3d, np.ones((len(skel3d), 1))))
        for idx, proj_each in enumerate(proj_mat):
            view_img = view_set[idx]
            if view_img.shape[0] > 1080:
                c_thinkness = 12
                l_thinkness = 12
            else:
                c_thinkness = 3
                l_thinkness = 1

            # let's reproject the 3d joint back to 2d view
            joint2d = np.dot(proj_each, skel3d_data.T).T
            joint2d_normalized_x = joint2d[:, 0] / joint2d[:, 2]
            joint2d_normalized_y = joint2d[:, 1] / joint2d[:, 2]
            j2d_x_list, j2d_y_list = [], []
            for jIdx in range(len(joint2d_normalized_y)):
                j2d_x = int(joint2d_normalized_x[jIdx] + 0.5)
                j2d_y = int(joint2d_normalized_y[jIdx] + 0.5)
                j2d_x_list.append(j2d_x)
                j2d_y_list.append(j2d_y)

            for jIdx, (j2d_x, j2d_y) in enumerate(zip(j2d_x_list, j2d_y_list)):
                if np.sum(skel3d[jIdx]) == 0.:
                    continue
                cv2.circle(view_img, (j2d_x, j2d_y), 6, skel_col_dict[skelId], thickness=c_thinkness, lineType=8,
                           shift=0)
            # draw limb
            if limb_map is not None and len(limb_map):
                for limb_a, limb_b in zip(limb_map[0], limb_map[1]):
                    j2d_a = (j2d_x_list[limb_a], j2d_y_list[limb_a])
                    j2d_b = (j2d_x_list[limb_b], j2d_y_list[limb_b])
                    if np.sum(skel3d[limb_a]) == 0. or np.sum(skel3d[limb_b]) == 0:
                        continue
                    if sum(j2d_a) > 0 and sum(j2d_b) > 0:
                        cv2.line(view_img, j2d_a, j2d_b, skel_col_dict[skelId], l_thinkness, 4)
    return view_set


def draw_points_for_each_view(point3d_set, proj_mat, views_path, view_set=None, color=None, thickness=3, reid=None):
    assert len(proj_mat) == len(views_path)
    if view_set is None:
        view_set = []
        for view_path_each in views_path:
            view_set.append(cv2.imread(view_path_each))
    assert len(view_set) == len(views_path)

    for pId, p3d in enumerate(point3d_set):
        # make sure data is not empty
        if not len(p3d):
            continue
        # make sure the correct shape
        if p3d.shape[1] != 3:
            p3d = p3d.T
        p3d_data = np.hstack((p3d, np.ones((len(p3d), 1))))

        if reid is not None:
            point_color = skel_col_dict[reid[pId] % len(skel_col_dict)]
        elif color is not None:
            point_color = color
        else:
            point_color = skel_col_dict[pId]

        for idx, proj_each in enumerate(proj_mat):
            view_img = view_set[idx]

            # let's reproject the 3d joint back to 2d view
            p2d = np.dot(proj_each, p3d_data.T).T
            p2d_normalized_x = p2d[:, 0] / p2d[:, 2]
            p2d_normalized_y = p2d[:, 1] / p2d[:, 2]
            for jIdx in range(len(p2d_normalized_y)):
                j2d_x = int(p2d_normalized_x[jIdx] + 0.5)
                j2d_y = int(p2d_normalized_y[jIdx] + 0.5)
                cv2.circle(view_img, (j2d_x, j2d_y), 2, point_color, thickness=thickness, lineType=8, shift=0)

    return view_set


def draw_points_for_each_view2(point3d_set, proj_mat, views_path, view_set=None, color_set=None, thickness=3):
    """
    :param point3d_set: [joint candidate, XYZ]
    :param color_set: color set for every joint candidate, [joint candidate, BGR]
    :return: view_set
    """
    assert len(proj_mat) == len(views_path)
    if view_set is None:
        view_set = []
        for view_path_each in views_path:
            view_set.append(cv2.imread(view_path_each))
    assert len(view_set) == len(views_path)

    # make sure data is not empty
    if not len(point3d_set):
        return view_set
    p3d_data = np.hstack((point3d_set, np.ones((len(point3d_set), 1))))
    if color_set is None:
        color_set = [(0, 0, 255) for _ in range(len(point3d_set))]
    for idx, proj_each in enumerate(proj_mat):
        view_img = view_set[idx]
        # let's reproject the 3d joint back to 2d view
        p2d = np.dot(proj_each, p3d_data.T).T
        p2d_normalized_x = p2d[:, 0] / p2d[:, 2]
        p2d_normalized_y = p2d[:, 1] / p2d[:, 2]
        for jIdx in range(len(p2d_normalized_y)):
            j2d_x = int(p2d_normalized_x[jIdx] + 0.5)
            j2d_y = int(p2d_normalized_y[jIdx] + 0.5)
            cv2.circle(view_img, (j2d_x, j2d_y), 2, color_set[jIdx], thickness=thickness, lineType=8, shift=0)

    return view_set


def draw_skel_3d(ax, skel_data, limb_map, color='r', jc_flg=False):
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_zlim((0.1, 3))
    plt.tight_layout()
    plt.ylim((-3, 3))
    plt.xlim((-3, 3))

    x = skel_data[:, 0].squeeze()
    y = skel_data[:, 1].squeeze()
    z = skel_data[:, 2].squeeze()
    # draw joint
    ax.scatter(x, y, z, c=color, s=10, marker='x')
    # draw limb
    if limb_map is not None:
        if jc_flg:
            return ax
        for limbIdx in range(len(limb_map[0])):
            jA_x = x[limb_map[0][limbIdx]]
            jA_y = y[limb_map[0][limbIdx]]
            jA_z = z[limb_map[0][limbIdx]]
            jB_x = x[limb_map[1][limbIdx]]
            jB_y = y[limb_map[1][limbIdx]]
            jB_z = z[limb_map[1][limbIdx]]
            limb_x = np.linspace(jA_x, jB_x, 100)
            limb_y = np.linspace(jA_y, jB_y, 100)
            limb_z = np.linspace(jA_z, jB_z, 100)
            ax.plot(limb_x, limb_y, limb_z, linewidth=1)
    return ax


def draw_seq_3d(seq_data, trans_mode=None, limb_map=None, jc_flg_list=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_list = ['r', 'b', 'g']
    if jc_flg_list is None:
        jc_flg_list = np.array([[False] for _ in range(seq_data)])

    for fIdx, frame_data in enumerate(seq_data):
        print("=> Frame {}".format(fIdx))
        plt.ion()
        plt.cla()

        for sIdx, skel_data in enumerate(frame_data):
            if len(skel_data) == 0:
                continue
            # draw skeleton
            if trans_mode == 'YZ':
                skel_data = np.array([skel_data[:, 0], skel_data[:, 2], skel_data[:, 1]]).T
            elif trans_mode == 'XY':
                skel_data = np.array([skel_data[:, 1], skel_data[:, 0], -skel_data[:, 2]]).T
            ax = draw_skel_3d(ax, skel_data, limb_map, color_list[sIdx % len(color_list)],
                              jc_flg=jc_flg_list[fIdx][sIdx])

        plt.pause(0.0001)
    plt.ioff()  # disable interactive mode
    plt.show()


def draw_seq_3d_in_one_scene(seq_data, limb_map=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_list = ['r', 'b', 'g']
    plt.ion()
    plt.cla()
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_zlim((0.1, 3))
    plt.tight_layout()
    plt.ylim((-3, 3))
    plt.xlim((-3, 3))

    for frame in seq_data:
        for bIdx, body in enumerate(frame):
            x = body[:, 0].squeeze()
            y = body[:, 1].squeeze()
            z = body[:, 2].squeeze()
            # draw joint
            ax.scatter(x, y, z, c=color_list[bIdx], s=10, marker='x')
            # draw limb
            if limb_map is not None:
                for limbIdx in range(len(limb_map[0])):
                    jA_x = x[limb_map[0][limbIdx]]
                    jA_y = y[limb_map[0][limbIdx]]
                    jA_z = z[limb_map[0][limbIdx]]
                    jB_x = x[limb_map[1][limbIdx]]
                    jB_y = y[limb_map[1][limbIdx]]
                    jB_z = z[limb_map[1][limbIdx]]
                    limb_x = np.linspace(jA_x, jB_x, 100)
                    limb_y = np.linspace(jA_y, jB_y, 100)
                    limb_z = np.linspace(jA_z, jB_z, 100)
                    ax.plot(limb_x, limb_y, limb_z, linewidth=1)

    plt.ioff()  # disable interactive mode
    plt.show()
