from intersection_plan import *
from line import *
from circle import *
import open3d as o3d

def read_points_file(file_name):
    points = np.loadtxt(file_name+'_pred_normals.txt', delimiter=";")
    edge_probability = np.loadtxt(file_name+'_pred_edge.txt')
    inst_num = np.loadtxt(file_name+'_pred_inst.txt').astype(int)
    return points, edge_probability, inst_num

def remove_outliers(points, edge_probability, inst_num):
    out_index = []
    new_vis = np.unique(inst_num, axis=0) # unique instance numbers
    for color in new_vis:
        index = np.where(inst_num == color)
        length = len(index[0])
        print("length: ", length)
        if length < 40:
            out_index = np.append(out_index, index)
    if len(out_index):
        out_index = out_index.astype(int)
    points_removed = np.delete(points, out_index, axis=0)
    edge_probability_removed = np.delete(edge_probability, out_index, axis=0)
    inst_num_removed = np.delete(inst_num, out_index, axis=0)
    return points_removed, edge_probability_removed, inst_num_removed

def get_edges_midpoint(gt_points, edge, inst, P, R):
    retrieval_radius = 0.04 # d = 3*d_bar
    inter_dict = {}

    edge_points = np.where(edge > P)
    e_list = edge_points[0]
    print(edge[edge_points].shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_points[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(gt_points[:, 3:6])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for e in e_list:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[e], retrieval_radius) # retrieval_radius
        neighborindex_list = idx[1:]
        C1 = np.array(gt_points[e][:3])
        N1 = np.array(gt_points[e][3:])

        for neighborindex in neighborindex_list:
            line_points = np.empty(shape=(0, 3))
            if(inst[e] != inst[neighborindex]):
                C2 = np.array(gt_points[neighborindex][:3])
                N2 = np.array(gt_points[neighborindex][3:])
                inter_p1, inter_p2 = get_intersection(C1, N1, C2, N2, R)
                if (True not in np.isnan(inter_p1)) and (True not in np.isnan(inter_p2)): # remove NAN
                    line_points = (inter_p1 + inter_p2)/2
                    line_points = np.reshape(line_points, (1, 3))
                line_points = line_points[[not np.all(line_points[i] == 0) for i in range(line_points.shape[0])], :]  # remove 0
                if line_points.shape[0] != 0:
                    inst_key = str(inst[e]) + '-' + str(inst[neighborindex])
                    if inst_key in inter_dict.keys():
                        inter_dict[inst_key] = np.append(inter_dict[inst_key], line_points, axis=0)
                    else:
                        inter_dict[inst_key] = line_points

    return inter_dict

def get_edges_linesample(gt_points, edge, inst, P, R):
    linesample = 10 # sample number
    retrieval_radius = 0.04 # d = 3*d_bar

    inter_dict = {}
    edge_points = np.where(edge > P)
    e_list = edge_points[0]
    print("edge[edge_points].shape", edge[edge_points].shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_points[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(gt_points[:, 3:6])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for e in e_list:
        print("e", e)
        [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[e], retrieval_radius) # retrieval_radius
        neighborindex_list = idx[1:]
        print("idx[1:]:", idx[1:])
        C1 = np.array(gt_points[e][:3])
        N1 = np.array(gt_points[e][3:])

        for neighborindex in neighborindex_list:
            line_points = np.empty(shape=(0, 3))
            if(inst[e] != inst[neighborindex]):
                C2 = np.array(gt_points[neighborindex][:3])
                N2 = np.array(gt_points[neighborindex][3:])
                inter_p1, inter_p2 = get_intersection(C1, N1, C2, N2, R)
                print("-------", C1, N1, C2, N2, R, inter_p1, "------", inter_p2, "-------------------------------")
                if (True not in np.isnan(inter_p1)) and (True not in np.isnan(inter_p2)): # remove NAN
                    X = np.linspace(start=inter_p1[0], stop=inter_p2[0], num=linesample).reshape(linesample, 1)
                    Y = np.linspace(start=inter_p1[1], stop=inter_p2[1], num=linesample).reshape(linesample, 1)
                    Z = np.linspace(start=inter_p1[2], stop=inter_p2[2], num=linesample).reshape(linesample, 1)
                    line_points = np.concatenate((X, Y, Z), axis=1)
                line_points = line_points[[not np.all(line_points[i] == 0) for i in range(line_points.shape[0])], :]  # remove 0
                if line_points.shape[0] != 0:
                    inst_key = str(inst[e]) + '-' + str(inst[neighborindex])
                    if inst_key in inter_dict.keys():
                        inter_dict[inst_key] = np.append(inter_dict[inst_key], line_points, axis=0)
                    else:
                        inter_dict[inst_key] = line_points
    return inter_dict

def find_max_distance(ind, points):
    max_distance = -999999
    farthest_point = 999999
    for i in range(points.shape[0]):
        if i != ind:
            length = math.sqrt(math.pow(abs(points[ind][0] - points[i][0]), 2) + math.pow(abs(points[ind][1] - points[i][1]), 2) + math.pow(abs(points[ind][2] - points[i][2]), 2))
            if length > max_distance:
                max_distance = length
                farthest_point = i

    return max_distance, farthest_point

def find_glob_max(points):
    glob_max = -999999
    glob_ind = 999999
    glob_i = 999999
    for i in range(points.shape[0]):
        p_max, f_p = find_max_distance(i, points)
        if p_max > glob_max:
            glob_max = p_max
            glob_ind = f_p
            glob_i = i
    return glob_i, glob_ind

def find_neighbor(ind, points_array):
    index = None
    p_xyzf = np.empty(shape=(0, 4))

    rows = points_array.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array[:, 0:3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[ind], rows)
    nei_list = idx[1:]
    for i in nei_list:
        if points_array[i][3] == 0:
            p_xyzf = points_array[i].reshape(1, 4)
            index = i
            points_array[i][3] = 1
            break
        else:
            continue
    return index, p_xyzf, points_array

def combine_key(inter_result):
    dic_combine = {}

    for key in list(inter_result.keys()):
        if key in inter_result:
            new_key = key.split("-")[1]+"-"+key.split("-")[0]
            if new_key in inter_result:
                new_value = np.append(inter_result[key], inter_result[new_key], axis=0)
                new_value = np.unique(new_value, axis=0)
                dic_combine[key] = new_value
                del inter_result[new_key]
            else:
                new_value = np.unique(inter_result[key], axis=0)
                dic_combine[key] = new_value
    print("combine keys(): ", len(dic_combine.keys()))
    print("combine keys num(): ", list(dic_combine.keys()))
    return dic_combine

def order_dic(dic_combine):
    dic_order = {}
    for key, value_unorder in dic_combine.items():
        ind1, ind2 = find_glob_max(value_unorder)
        rows = value_unorder.shape[0]
        flag_col = np.zeros(rows)
        value_unorder_flag = np.insert(value_unorder, 3, flag_col, axis=1)
        value_unorder_flag[ind1][3] = 1
        value_order_flag = value_unorder_flag[ind1].reshape(1, 4)

        times = rows - 1
        for t in range(times):
            n_index, xyzf, points_array_new = find_neighbor(ind1, value_unorder_flag)
            ind1 = n_index
            value_unorder_flag = points_array_new
            value_order_flag = np.append(value_order_flag, xyzf, axis=0)
            if n_index == ind2:
                break
        dic_order[key] = value_order_flag[:, 0:3]
    return dic_order

def save_dic_allpoints(dic_order, file_num, select_edges_p, intersection_radius):
    inter = np.empty(shape=(0, 3))
    for key, value in dic_order.items():
        inter = np.append(inter, value, axis=0)
    np.savetxt(str(file_num)+"res-p"+str(select_edges_p)+"-r"+str(intersection_radius)
                   +"_allpoints.txt", inter, fmt="%.6f")

def save_dic_linepoints(dic_order, file_num, select_edges_p, intersection_radius):
    filename = str(file_num)+"res-p"+str(select_edges_p)+"-r"+str(intersection_radius)+"_linepoints.txt"
    with open(filename, 'w') as f:
        print("keys(): ", len(dic_order.keys()))
        f.write(str(len(dic_order.keys()))+"\n")
        for _, value1 in dic_order.items():
            f.write(str(value1.shape[0])+"\n")
            print("value.shape[0]: ", value1.shape[0])
            for row in range(len(value1)):
                for col in range(3):
                    f.write("%.6f%c"%(value1[row][col], " \n"[col == 2]))

def remove_outlier(points, nb_neighbors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    res = pcd.remove_statistical_outlier(nb_neighbors, 0.5)
    new_points = points[res[1]]
    return new_points


if __name__ == '__main__':

    file_num = "./data"
    select_edges_p = 0.7 # edge point threshold
    intersection_radius = 0.04 # d = 3*d_bar

    gt_points, edge, inst = read_points_file(str(file_num))
    gt_points, edge, inst = remove_outliers(gt_points, edge, inst)
    inter_result = get_edges_linesample(gt_points, edge, inst, select_edges_p, intersection_radius)
    mid_inter_result = get_edges_midpoint(gt_points, edge, inst, select_edges_p, intersection_radius)
    inter_dic = combine_key(inter_result)
    mid_inter_dic = combine_key(mid_inter_result)
    save_dic_allpoints(inter_dic, file_num + "_dic_", select_edges_p, intersection_radius)

    c_fit_dic = {}
    nonc_dic = {}
    l_fit_dic = {}
    b_dic = {}

    for key, value in inter_dic.items():
        print("=====fi-check-circle=====", key)
        value_temp = value
        value = remove_outlier(value, 10)
        circle, circle_center, radius, c_residuals, normal, u = circle_segmentation(np.array(value))
        if (c_residuals.size == 0):
            pass
        elif (c_residuals[0] > 0.5): # not a circle
            print("c_residuals -> non-circle", c_residuals)
            nonc_dic[key] = value_temp
        else:
            p2p_dis = get_r_ball(circle)
            t = find_arc(value, circle, p2p_dis)
            t_rad = deg2rad(t)
            fitcircle = sample_circle(t_rad, circle_center, radius, normal, u)
            c_fit_dic[key] = fitcircle

    save_dic_allpoints(c_fit_dic, file_num+"_circle_fit_", select_edges_p, intersection_radius)
    print("=====circle_fit-over!=====")

    for key, value in nonc_dic.items():
        print("=====se-check-line=====", key)
        value_ori = value
        k1, b1, k2, b2, l_residuals = linear_fitting_3D_points(np.array(value))
        print("l_residuals", l_residuals)
        if (l_residuals > 0.00009): # not a line
            print("l_residuals -> b-spline", l_residuals)
            mid_value = mid_inter_dic[key]
            value_temp = remove_outlier(mid_value, 5)
            b_dic[key] = value_temp
        else:
            ran_seed = random.randint(0, value.shape[0] - 1)
            z = value[ran_seed, 2]
            z0, z1 = find_start_end(value, k1, b1, k2, b2, z)
            line_points = sample_line_new(z0, z1)
            l_fit_dic[key] = line_points

    save_dic_allpoints(l_fit_dic, file_num+"_line_fit_", select_edges_p, intersection_radius)
    print("=====line_fit-over!=====")

    # bezier curve
    b_dic_order = order_dic(b_dic)
    save_dic_linepoints(b_dic_order, file_num+"_b_", select_edges_p, intersection_radius)