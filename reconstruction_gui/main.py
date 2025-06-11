import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from intersection_plan import *
from line import *
from circle import *
from nor_opt import *
from seg_layout import SegLayout
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matlab.engine
import pandas as pd

def adjust_brightness(rgb_color, factor):
    adjusted_color = np.clip(rgb_color * factor / 255.0, 0, 1.0)
    return adjusted_color

def normals_lineset(points_pred_normals, normal_scale = 0.03, color = [0.41, 0.41, 0.41]):
    line_set = o3d.geometry.LineSet()
    start = points_pred_normals[:, 0:3]
    end = points_pred_normals[:, 0:3] + (points_pred_normals[:, 3:6] * normal_scale)
    points = np.concatenate((start, end))
    line_set.points = o3d.utility.Vector3dVector(points)
    size = len(start)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray([[i, i+size] for i in range(0, size)]))
    line_set.paint_uniform_color(color)
    return line_set

def combine_key(inter_result):
    dic_combine = {}
    for key in list(inter_result.keys()):
        if key in inter_result:
            new_key = key.split("-")[1] + "-" + key.split("-")[0]
            if new_key in inter_result:
                new_value = np.append(inter_result[key]["points"], inter_result[new_key]["points"], axis=0)
                new_value = np.unique(new_value, axis=0)
                dic_combine[key] = {}
                dic_combine[key]["points"] = new_value
                dic_combine[key]["probability"] = inter_result[key]["probability"]
                dic_combine[key]["retrieval_radius"] = inter_result[key]["retrieval_radius"]
                dic_combine[key]["intersecting_radius"] = inter_result[key]["intersecting_radius"]
                dic_combine[key]["inter_endpoints"] = np.append(inter_result[key]["inter_endpoints"], inter_result[new_key]["inter_endpoints"], axis=0)
                del inter_result[new_key]
                # print(key)
            else:
                new_value = np.unique(inter_result[key]["points"], axis=0)
                dic_combine[key] = {}
                dic_combine[key]["points"] = new_value
                dic_combine[key]["probability"] = inter_result[key]["probability"]
                dic_combine[key]["retrieval_radius"] = inter_result[key]["retrieval_radius"]
                dic_combine[key]["intersecting_radius"] = inter_result[key]["intersecting_radius"]
                dic_combine[key]["inter_endpoints"] = inter_result[key]["inter_endpoints"]
    # add color to each key
    for index, key in enumerate(dic_combine):
        color = [random.randint(0, 255) for _ in range(3)]
        dic_combine[key]["color"] = color

    print("combine keys(): ", len(dic_combine.keys()))
    print("combine keys num(): ", list(dic_combine.keys()))

    return dic_combine

def midpoints_combine_key(inter_result):
    dic_combine = {}
    for key in list(inter_result.keys()):
        if key in inter_result:
            new_key = key.split("-")[1] + "-" + key.split("-")[0]
            if new_key in inter_result:
                new_value = np.append(inter_result[key], inter_result[new_key], axis=0)
                new_value = np.unique(new_value, axis=0)
                dic_combine[key] = new_value
                del inter_result[new_key]
                # print(key)
            else:
                new_value = np.unique(inter_result[key], axis=0)
                dic_combine[key] = new_value
    print("combine keys(): ", len(dic_combine.keys()))
    print("combine keys num(): ", list(dic_combine.keys()))
    return dic_combine

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

class App:

    count = 0
    MENU_OPEN = 1
    MENU_SHOW_POINT_CLOUDS = 5
    MENU_SHOW_EDGE_POINTS = 6
    MENU_SHOW_VIS_SEGMENTS = 7
    MENU_QUIT = 20
    MENU_ABOUT = 21
    show_point_clouds = True
    show_vis_segments = True
    show_edge_points = True

    clean_segments_check_value = False
    p_switch_value = False
    retrieval_r_check_value = True
    intersecting_r_check_value = False
    circle_rm_out_check_value = False
    circle_r_ball_check_value = False
    circle_n_gap_check_value = False
    circle_residuals_threshold_check_value = False
    line_rm_out_check_value = False
    line_residuals_threshold_check_value = False
    visualize_order_points_check_value = True
    bspline_rm_out_check_value = False

    do_editing_points = False

    count_key_temp = 0
    segments_key_list_temp = []
    segments_key_label_dic = {}
    count_key_temp_c = 0

    def __init__(self):
        # initialize the application
        gui.Application.instance.initialize()
        # create a window
        self.window = gui.Application.instance.create_window('My Edge', 1760, 990)
        w = self.window

        self.pcd_name = None

        # create a scene widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        self._init_Load_pannel()
        self._init_N_pannel()
        self._init_E_pannel()
        self._init_Linear_Segments_pannel()
        self._init_Segments_Key_pannel()
        self._init_Fit_Circle_pannel()
        self._init_Fit_Line_pannel()
        self._init_Fit_BSpline_pannel()
        self._init_Pick_Points_pannel()
        self._init_Normal_Optimization_pannel()

        # layout the scene widget
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._Load_pannel)
        self.window.add_child(self._N_pannel)
        self.window.add_child(self._E_pannel)
        self.window.add_child(self._Linear_Segments_pannel)
        self.window.add_child(self._Segments_Key_pannel)
        self.window.add_child(self._Fit_Circle_pannel)
        self.window.add_child(self._Fit_Line_pannel)
        self.window.add_child(self._Fit_BSpline_pannel)
        self.window.add_child(self._Pick_Points_pannel)
        self.window.add_child(self._Normal_Optimization_pannel)


        # start matlab
        self.engine = matlab.engine.start_matlab()
        self.engine.eval("feature('DefaultCharacterSet', 'UTF-8')")

        # only for menu bar in Windows
        if gui.Application.instance.menubar is None:
            # file menu
            file_menu = gui.Menu()
            file_menu.add_item("Open", App.MENU_OPEN)
            file_menu.add_separator()
            file_menu.add_item("Quit", App.MENU_QUIT)
            # show menu
            show_menu = gui.Menu()
            show_menu.add_item("Show Point Clouds", App.MENU_SHOW_POINT_CLOUDS)
            show_menu.add_separator()
            show_menu.add_item("Show Vis Segments", App.MENU_SHOW_VIS_SEGMENTS)
            show_menu.add_separator()
            show_menu.add_item("Show Edge Points", App.MENU_SHOW_EDGE_POINTS)
            show_menu.set_checked(App.MENU_SHOW_POINT_CLOUDS, True)
            show_menu.set_checked(App.MENU_SHOW_VIS_SEGMENTS, True)
            show_menu.set_checked(App.MENU_SHOW_EDGE_POINTS, True)
            # help menu
            help_menu = gui.Menu()
            help_menu.add_item("About", App.MENU_ABOUT)
            help_menu.set_enabled(App.MENU_ABOUT, False)
            # menu bar
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Show", show_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu
            # register menu events
            w.set_on_menu_item_activated(App.MENU_OPEN, self._menu_open)
            w.set_on_menu_item_activated(App.MENU_QUIT, self._menu_quit)
            w.set_on_menu_item_activated(App.MENU_SHOW_POINT_CLOUDS, self._menu_show_point_clouds)
            w.set_on_menu_item_activated(App.MENU_SHOW_VIS_SEGMENTS, self._menu_show_vis_segments)
            w.set_on_menu_item_activated(App.MENU_SHOW_EDGE_POINTS, self._menu_show_edge_points)

    def _init_Normal_Optimization_pannel(self):
        w = self.window
        em = w.theme.font_size  # size relative to the theme
        # right side normal optimization panel
        self._Normal_Optimization_pannel = gui.CollapsableVert('Normal_Optimization_Pannel', 0.2 * em)

        self.plane_threshold_textedit = gui.TextEdit()
        self.plane_threshold_textedit.text_value = str(0.5)
        self.plane_threshold_check = gui.Checkbox("")
        self.plane_threshold_check.set_on_checked(self.plane_threshold_check_fun)

        self.plane_threshold_layout = gui.Horiz()
        self.plane_threshold_layout.add_stretch()
        self.plane_threshold_layout.add_child(gui.Label(" Plane Threshold "))
        self.plane_threshold_layout.add_stretch()
        self.plane_threshold_layout.add_child(self.plane_threshold_textedit)
        self.plane_threshold_layout.add_stretch()
        self.plane_threshold_layout.add_child(self.plane_threshold_check)

        self._Normal_Optimization_pannel.add_child(self.plane_threshold_layout)

    def _init_Load_pannel(self):
        w = self.window
        em = w.theme.font_size  # size relative to the theme
        # right side load panel
        self._Load_pannel = gui.CollapsableVert('Load_Pannel', 0.2 * em)

        # clean segments
        clean_segments_move_slider = gui.Slider(gui.Slider.INT)
        clean_segments_move_slider.set_limits(1, 100)
        clean_segments_move_slider.int_value = 40
        self.clean_segments_val = clean_segments_move_slider.int_value
        clean_segments_move_slider.set_on_value_changed(self.clean_segments_slider_fun)
        self.clean_segments_check = gui.Checkbox("")
        self.clean_segments_check.set_on_checked(self.clean_segments_check_fun)

        clean_segments_layout = gui.Horiz()
        clean_segments_layout.add_stretch()
        clean_segments_layout.add_child(gui.Label(' Clean Segments '))
        clean_segments_layout.add_stretch()
        clean_segments_layout.add_child(clean_segments_move_slider)
        clean_segments_layout.add_stretch()
        clean_segments_layout.add_child(self.clean_segments_check)

        self._Load_pannel.add_child(clean_segments_layout)

    def _init_E_pannel(self):
        w = self.window
        em = w.theme.font_size # size relative to the theme
        # right side edge points panel
        self._E_pannel = gui.CollapsableVert('Edge_Points_Pannel', 0.2*em)

        move_slider = gui.Slider(gui.Slider.DOUBLE)
        move_slider.set_limits(0.5, 1.0)
        move_slider.double_value = 0.80
        move_slider.set_on_value_changed(self._slider_test)

        self._p_switch = gui.Checkbox("")
        self._p_switch.set_on_checked(self.p_check_fun)

        P_slider_layout = gui.Horiz()
        P_slider_layout.add_stretch()
        P_slider_layout.add_child(gui.Label(' Probability '))
        P_slider_layout.add_stretch()
        P_slider_layout.add_child(move_slider)
        P_slider_layout.add_stretch()
        P_slider_layout.add_child(self._p_switch)

        self._E_pannel.add_child(P_slider_layout)

    def _init_N_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side normal panel
        self._N_pannel = gui.CollapsableVert('Normal_Pannel', 0.2*em)

        self._n_switch = gui.ToggleSwitch("")
        self._n_switch.set_on_clicked(self.n_toggle)
        self.n_switch_edit = gui.TextEdit()

        N_layout = gui.Horiz()
        N_layout.add_stretch()
        N_layout.add_child(self.n_switch_edit)
        N_layout.add_stretch()
        N_layout.add_child(self._n_switch)

        self._N_pannel.add_child(N_layout)

    def _init_Linear_Segments_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side linear segments panel
        self._Linear_Segments_pannel = gui.CollapsableVert('Linear_Segments_Pannel', 0.2*em)

        # retrieval radius
        retrieval_r_move_slider = gui.Slider(gui.Slider.DOUBLE)
        retrieval_r_move_slider.set_limits(0.001, 0.060)
        retrieval_r_move_slider.double_value = 0.04
        self.retrieval_r_val = retrieval_r_move_slider.double_value
        retrieval_r_move_slider.set_on_value_changed(self.retrieval_r_slider_fun)

        self._retrieval_r_check = gui.Checkbox("")
        self._retrieval_r_check.checked = True
        self._retrieval_r_check.set_on_checked(self.retrieval_r_check_fun)

        retrieval_r_layout = gui.Horiz()
        retrieval_r_layout.add_stretch()
        retrieval_r_layout.add_child(gui.Label(' Retrieval Radius '))
        retrieval_r_layout.add_stretch()
        retrieval_r_layout.add_child(retrieval_r_move_slider)
        retrieval_r_layout.add_stretch()
        retrieval_r_layout.add_child(self._retrieval_r_check)

        # intersecting radius
        intersecting_r_move_slider = gui.Slider(gui.Slider.DOUBLE)
        intersecting_r_move_slider.set_limits(0.01, 0.05)
        intersecting_r_move_slider.double_value = 0.030
        self.intersecting_r_val = intersecting_r_move_slider.double_value
        intersecting_r_move_slider.set_on_value_changed(self.intersecting_r_slider_fun)

        self._intersecting_r_check = gui.Checkbox("")
        self._intersecting_r_check.set_on_checked(self.intersecting_r_check_fun)

        intersecting_r_layout = gui.Horiz()
        intersecting_r_layout.add_stretch()
        intersecting_r_layout.add_child(gui.Label(' Intersecting Radius '))
        intersecting_r_layout.add_stretch()
        intersecting_r_layout.add_child(intersecting_r_move_slider)
        intersecting_r_layout.add_stretch()
        intersecting_r_layout.add_child(self._intersecting_r_check)

        linear_segments_start_button = gui.Button("Start Generating Linear Segments")
        linear_segments_start_button.vertical_padding_em = 0.08
        linear_segments_start_button.set_on_clicked(self.linear_segments_start_fun)


        self._Linear_Segments_pannel.add_child(retrieval_r_layout)
        self._Linear_Segments_pannel.add_child(intersecting_r_layout)
        self._Linear_Segments_pannel.add_child(linear_segments_start_button)

    def _init_Fit_Circle_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side circle fitting panel
        self._Fit_Circle_pannel = gui.CollapsableVert('Fit_Circle_Pannel', 0.2*em)

        # check whether to remove outliers
        self.use_circle_rm_out_check = gui.Checkbox("")
        self.use_circle_rm_out_check.checked = False
        self.use_circle_rm_out_check.set_on_checked(self.use_circle_rm_out_check_fun)
        use_circle_rm_out_layout = gui.Horiz()
        use_circle_rm_out_layout.add_stretch()
        use_circle_rm_out_layout.add_child(gui.Label(" Use to Remove Outliers? "))
        use_circle_rm_out_layout.add_stretch()
        use_circle_rm_out_layout.add_child(self.use_circle_rm_out_check)

        # outliers removal parameter
        circle_rm_out_move_slider = gui.Slider(gui.Slider.INT)
        circle_rm_out_move_slider.set_limits(1, 30)
        circle_rm_out_move_slider.int_value = 20
        self.circle_rm_out_val = circle_rm_out_move_slider.int_value
        circle_rm_out_move_slider.set_on_value_changed(self.circle_rm_out_slider_fun)

        self._circle_rm_out_check = gui.Checkbox("")
        self._circle_rm_out_check.set_on_checked(self.circle_rm_out_check_fun)

        self.circle_rm_out_layout = gui.Horiz()
        self.circle_rm_out_layout.add_stretch()
        self.circle_rm_out_layout.add_child(circle_rm_out_move_slider)
        self.circle_rm_out_layout.add_stretch()
        self.circle_rm_out_layout.add_child(self._circle_rm_out_check)
        self.circle_rm_out_layout.visible = False

        # r_ball = p2p_dis * 5000
        self.circle_r_ball_textedit = gui.TextEdit()
        self.circle_r_ball_check = gui.Checkbox("")
        self.circle_r_ball_check.set_on_checked(self.circle_r_ball_check_fun)
        self.circle_r_ball_layout = gui.Horiz()
        self.circle_r_ball_layout.add_stretch()
        self.circle_r_ball_layout.add_child(gui.Label(" Ball Radius Search "))
        self.circle_r_ball_layout.add_stretch()
        self.circle_r_ball_layout.add_child(self.circle_r_ball_textedit)
        self.circle_r_ball_textedit.text_value = str(5000)
        self.circle_r_ball_layout.add_stretch()
        self.circle_r_ball_layout.add_child(self.circle_r_ball_check)

        # circle fit parameter
        circle_n_gap_move_slider = gui.Slider(gui.Slider.INT)
        circle_n_gap_move_slider.set_limits(1, 400)
        circle_n_gap_move_slider.int_value = 36
        self.circle_n_gap_val = circle_n_gap_move_slider.int_value
        circle_n_gap_move_slider.set_on_value_changed(self.circle_n_gap_slider_fun)
        self.circle_n_gap_check = gui.Checkbox("")
        self.circle_n_gap_check.set_on_checked(self.circle_n_gap_check_fun)

        self.circle_n_gap_layout = gui.Horiz()
        self.circle_n_gap_layout.add_child(gui.Label(" Gap in the Arc "))
        self.circle_n_gap_layout.add_stretch()
        self.circle_n_gap_layout.add_child(circle_n_gap_move_slider)
        self.circle_n_gap_layout.add_stretch()
        self.circle_n_gap_layout.add_child(self.circle_n_gap_check)

        # circle fit residuals threshold
        circle_residuals_threshold_move_slider = gui.Slider(gui.Slider.DOUBLE)
        circle_residuals_threshold_move_slider.set_limits(0.0, 0.5)
        circle_residuals_threshold_move_slider.double_value = 0.30
        self.circle_residuals_threshold_val = circle_residuals_threshold_move_slider.double_value
        circle_residuals_threshold_move_slider.set_on_value_changed(self.circle_residuals_threshold_slider_fun)
        self.circle_residuals_threshold_check = gui.Checkbox("")
        self.circle_residuals_threshold_check.set_on_checked(self.circle_residuals_threshold_check_fun)

        self.circle_residuals_threshold_layout = gui.Horiz()
        self.circle_residuals_threshold_layout.add_child(gui.Label(" Residuals Threshold "))
        self.circle_residuals_threshold_layout.add_stretch()
        self.circle_residuals_threshold_layout.add_child(circle_residuals_threshold_move_slider)
        self.circle_residuals_threshold_layout.add_stretch()
        self.circle_residuals_threshold_layout.add_child(self.circle_residuals_threshold_check)

        # start button
        circle_fit_start_button = gui.Button("Start Generating Circles")
        circle_fit_start_button.vertical_padding_em = 0.08
        circle_fit_start_button.set_on_clicked(self.circle_fit_start_fun)

        self._Fit_Circle_pannel.add_child(use_circle_rm_out_layout)
        self._Fit_Circle_pannel.add_child(self.circle_rm_out_layout)
        self._Fit_Circle_pannel.add_child(self.circle_r_ball_layout)
        self._Fit_Circle_pannel.add_child(self.circle_n_gap_layout)
        self._Fit_Circle_pannel.add_child(self.circle_residuals_threshold_layout)
        self._Fit_Circle_pannel.add_child(circle_fit_start_button)

    def _init_Fit_Line_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side line fitting panel
        self._Fit_Line_pannel = gui.CollapsableVert('Fit_Line_Pannel', 0.2*em)

        # check whether to remove outliers
        self.use_line_rm_out_check = gui.Checkbox("")
        self.use_line_rm_out_check.checked = False
        self.use_line_rm_out_check.set_on_checked(self.use_line_rm_out_check_fun)
        use_line_rm_out_layout = gui.Horiz()
        use_line_rm_out_layout.add_stretch()
        use_line_rm_out_layout.add_child(gui.Label(" Use to Remove Outliers? "))
        use_line_rm_out_layout.add_stretch()
        use_line_rm_out_layout.add_child(self.use_line_rm_out_check)

        # remove outliers parameter
        line_rm_out_move_slider = gui.Slider(gui.Slider.INT)
        line_rm_out_move_slider.set_limits(1, 30)
        line_rm_out_move_slider.int_value = 20
        self.line_rm_out_val = line_rm_out_move_slider.int_value
        line_rm_out_move_slider.set_on_value_changed(self.line_rm_out_slider_fun)
        self._line_rm_out_check = gui.Checkbox("")
        self._line_rm_out_check.set_on_checked(self.line_rm_out_check_fun)

        self.line_rm_out_layout = gui.Horiz()
        self.line_rm_out_layout.add_stretch()
        self.line_rm_out_layout.add_child(line_rm_out_move_slider)
        self.line_rm_out_layout.add_stretch()
        self.line_rm_out_layout.add_child(self._line_rm_out_check)
        self.line_rm_out_layout.visible = False

        # line fit residuals threshold
        line_residuals_threshold_move_slider = gui.Slider(gui.Slider.DOUBLE)
        line_residuals_threshold_move_slider.set_limits(0.000, 0.010)
        line_residuals_threshold_move_slider.double_value = 0.0005
        self.line_residuals_threshold_val = line_residuals_threshold_move_slider.double_value
        line_residuals_threshold_move_slider.set_on_value_changed(self.line_residuals_threshold_slider_fun)
        self.line_residuals_threshold_check = gui.Checkbox("")
        self.line_residuals_threshold_check.set_on_checked(self.line_residuals_threshold_check_fun)

        self.line_residuals_threshold_layout = gui.Horiz()
        self.line_residuals_threshold_layout.add_child(gui.Label(" Residuals Threshold "))
        self.line_residuals_threshold_layout.add_stretch()
        self.line_residuals_threshold_layout.add_child(line_residuals_threshold_move_slider)
        self.line_residuals_threshold_layout.add_stretch()
        self.line_residuals_threshold_layout.add_child(self.line_residuals_threshold_check)

        # start button
        line_fit_start_button = gui.Button("Start Generating Lines")
        line_fit_start_button.vertical_padding_em = 0.08
        line_fit_start_button.set_on_clicked(self.line_fit_start_fun)

        self._Fit_Line_pannel.add_child(use_line_rm_out_layout)
        self._Fit_Line_pannel.add_child(self.line_rm_out_layout)
        self._Fit_Line_pannel.add_child(self.line_residuals_threshold_layout)
        self._Fit_Line_pannel.add_child(line_fit_start_button)

    def _init_Fit_BSpline_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side bspline fitting panel
        self._Fit_BSpline_pannel = gui.CollapsableVert('Fit_BSpline_Pannel', 0.2 * em)

        # check whether to remove outliers
        self.use_bspline_rm_out_check = gui.Checkbox("")
        self.use_bspline_rm_out_check.checked = False
        self.use_bspline_rm_out_check.set_on_checked(self.use_bspline_rm_out_check_fun)
        use_bspline_rm_out_layout = gui.Horiz()
        use_bspline_rm_out_layout.add_stretch()
        use_bspline_rm_out_layout.add_child(gui.Label(" Use to Remove Outliers? "))
        use_bspline_rm_out_layout.add_stretch()
        use_bspline_rm_out_layout.add_child(self.use_bspline_rm_out_check)

        # remove outliers parameter
        bspline_rm_out_move_slider = gui.Slider(gui.Slider.INT)
        bspline_rm_out_move_slider.set_limits(1, 30)
        bspline_rm_out_move_slider.int_value = 5
        self.bspline_rm_out_val = bspline_rm_out_move_slider.int_value
        bspline_rm_out_move_slider.set_on_value_changed(self.bspline_rm_out_slider_fun)
        self._bspline_rm_out_check = gui.Checkbox("")
        self._bspline_rm_out_check.set_on_checked(self.bspline_rm_out_check_fun)

        self.bspline_rm_out_layout = gui.Horiz()
        self.bspline_rm_out_layout.add_stretch()
        self.bspline_rm_out_layout.add_child(bspline_rm_out_move_slider)
        self.bspline_rm_out_layout.add_stretch()
        self.bspline_rm_out_layout.add_child(self._bspline_rm_out_check)
        self.bspline_rm_out_layout.visible = False

        order_points_start_button = gui.Button("Start Sorting Points")
        order_points_start_button.vertical_padding_em = 0.08
        order_points_start_button.set_on_clicked(self.order_points_start_fun)
        self.visualize_order_points_check = gui.Checkbox("")
        self.visualize_order_points_check.checked = self.visualize_order_points_check_value
        self.visualize_order_points_check.set_on_checked(self.visualize_order_points_check_fun)

        self.order_points_layout = gui.Horiz()
        self.order_points_layout.add_child(order_points_start_button)
        self.order_points_layout.add_stretch()
        self.order_points_layout.add_child(gui.Label("Vis Order"))
        self.order_points_layout.add_child(self.visualize_order_points_check)

        bspline_fit_start_button = gui.Button("Start Generating Bspline")
        bspline_fit_start_button.vertical_padding_em = 0.08
        bspline_fit_start_button.set_on_clicked(self.bspline_fit_start_fun)

        self._Fit_BSpline_pannel.add_child(use_bspline_rm_out_layout)
        self._Fit_BSpline_pannel.add_child(self.bspline_rm_out_layout)
        self._Fit_BSpline_pannel.add_child(self.order_points_layout)
        self._Fit_BSpline_pannel.add_child(bspline_fit_start_button)

    def _init_Pick_Points_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side pick points panel
        self._Pick_Points_pannel = gui.CollapsableVert('Pick_Points_Pannel', 0.2 * em)

        self.seg_key_textedit = gui.TextEdit()
        self.seg_or_fit_combobox = gui.Combobox()
        self.seg_or_fit_combobox.add_item("Seg")
        self.seg_or_fit_combobox.add_item("Fit")
        self.seg_or_fit_combobox.selected_text = "Seg"
        self.seg_or_fit_text = self.seg_or_fit_combobox.selected_text

        self.seg_or_fit_combobox.set_on_selection_changed(self.seg_or_fit_combo_fun)
        open_button = gui.Button("Open")
        open_button.vertical_padding_em = 0.08
        open_button.set_on_clicked(self.open_fun)
        close_button = gui.Button("Close")
        close_button.vertical_padding_em = 0.08
        close_button.set_on_clicked(self.close_fun)

        remove_button = gui.Button("Remove")
        remove_button.vertical_padding_em = 0.08
        remove_button.set_on_clicked(self.remove_fun)
        invert_button = gui.Button("Invert")
        invert_button.vertical_padding_em = 0.08
        invert_button.set_on_clicked(self.invert_fun)
        self.pick_points_layout_r_i = gui.Horiz()
        self.pick_points_layout_r_i.add_child(remove_button)
        self.pick_points_layout_r_i.add_stretch()
        self.pick_points_layout_r_i.add_child(invert_button)

        self.pick_points_layout = gui.Horiz()
        self.pick_points_layout.add_child(self.seg_key_textedit)
        self.pick_points_layout.add_stretch()
        self.pick_points_layout.add_child(self.seg_or_fit_combobox)
        self.pick_points_layout.add_stretch()
        self.pick_points_layout.add_child(open_button)
        self.pick_points_layout.add_child(close_button)

        self._Pick_Points_pannel.add_child(self.pick_points_layout)
        self._Pick_Points_pannel.add_child(self.pick_points_layout_r_i)


    def _init_Segments_Key_pannel(self):
        w = self.window
        em = w.theme.font_size
        # right side segments key panel
        self._Segments_Key_pannel = gui.CollapsableVert('Segments_Key_Pannel', 0.05*em)

        # all segments control
        self.lock_all_check = gui.Checkbox("")
        self.lock_all_check.set_on_checked(self.lock_all_check_fun)
        self.lock_count_textedit = gui.TextEdit()
        self.visualize_all_segments_check = gui.Checkbox("")
        self.visualize_all_segments_check.set_on_checked(self.visualize_all_segments_check_fun)
        self.visualize_all_fitting_check = gui.Checkbox("")
        self.visualize_all_fitting_check.set_on_checked(self.visualize_all_fitting_check_fun)
        save_all_segments_button = gui.Button(" Save All ")
        save_all_segments_button.vertical_padding_em = 0.04
        save_all_segments_button.set_on_clicked(self.save_all_segments_fun)
        save_all_fitting_button = gui.Button(" Save Locked ")
        save_all_fitting_button.vertical_padding_em = 0.04
        save_all_fitting_button.set_on_clicked(self.save_all_fitting_fun)

        self.all_control_layout = gui.Horiz()
        self.all_control_layout.add_child(self.lock_all_check)
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(self.lock_count_textedit)
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(gui.Label(" All Seg"))
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(self.visualize_all_segments_check)
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(save_all_segments_button)
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(gui.Label(" All Fit"))
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(self.visualize_all_fitting_check)
        self.all_control_layout.add_stretch()
        self.all_control_layout.add_child(save_all_fitting_button)

        self._Segments_Key_pannel.add_child(self.all_control_layout)

        # create 100 segment layouts
        self.seg_layouts = []
        for i in range(0, 100):
            seg_layout_instance = SegLayout(self)
            self.seg_layouts.append(seg_layout_instance)
            self._Segments_Key_pannel.add_child(seg_layout_instance.seg_layout)

    # open file dialog
    def _menu_open(self):
        file_picker = gui.FileDialog(gui.FileDialog.OPEN, "Select file...", self.window.theme)

        file_picker.add_filter('.txt', 'points_pred_normals files')
        file_picker.add_filter('', 'All files')

        file_picker.set_path('./data')

        file_picker.set_on_cancel(self._on_cancel)
        file_picker.set_on_done(self._on_done)

        self.window.show_dialog(file_picker)

    def _on_cancel(self):

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        os.chdir(current_directory)

        self.window.close_dialog()

    def _on_done(self, filename):

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        os.chdir(current_directory)

        self.window.close_dialog()
        self.load_scene(filename)

    def load_scene(self, path):
        self._scene.scene.clear_geometry()
        filename = os.path.basename(path)
        self.pcd_path = os.path.dirname(path)
        self.pcd_name = filename[:-17] # pcd_name only use 8 characters

    def load_xyz(self, points):
        if self._scene.scene.has_geometry(self.pcd_name + "_points"):
            self._scene.scene.remove_geometry(self.pcd_name + "_points")

        self.pcd_points = points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
        pcd.paint_uniform_color([1, 0, 1])

        material = rendering.MaterialRecord()
        material.point_size = 5

        material.shader = 'defaultUnlit'
        self._scene.scene.add_geometry(self.pcd_name + "_points", pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_points", self.show_point_clouds)
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

        self.load_n_xyz()

    # quit the application
    def _menu_quit(self):
        self.engine.quit()
        self.window.close()

    # show or hide point clouds
    def _menu_show_point_clouds(self):
        self.show_point_clouds = not self.show_point_clouds
        gui.Application.instance.menubar.set_checked(App.MENU_SHOW_POINT_CLOUDS, self.show_point_clouds)
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + "_points"):
                self._scene.scene.show_geometry(self.pcd_name + "_points", self.show_point_clouds)
            else:
                self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def _menu_show_vis_segments(self):
        self.show_vis_segments = not self.show_vis_segments
        gui.Application.instance.menubar.set_checked(App.MENU_SHOW_VIS_SEGMENTS, self.show_vis_segments)
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + "_pred_Vis_I"):
                self._scene.scene.show_geometry(self.pcd_name + "_pred_Vis_I", self.show_vis_segments)
            else:
                self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def _menu_show_edge_points(self):
        self.show_edge_points = not self.show_edge_points
        gui.Application.instance.menubar.set_checked(App.MENU_SHOW_EDGE_POINTS, self.show_edge_points)
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + "_pred_edge"):
                self._scene.scene.show_geometry(self.pcd_name + "_pred_edge", self.show_edge_points)
            else:
                self.window.show_message_box('ERROR', 'Please use the Edge_Points_Pannel first!')
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r

        Normal_Optimization_pannel_width = 20 * layout_context.theme.font_size
        Normal_Optimization_pannel_height = min(
            r.height, self._Normal_Optimization_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Normal_Optimization_pannel.frame = gui.Rect(r.x, r.y, Normal_Optimization_pannel_width, Normal_Optimization_pannel_height)

        Load_pannel_width = 20 * layout_context.theme.font_size
        Load_pannel_height = min(
            r.height, self._Load_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Load_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height, Load_pannel_width, Load_pannel_height)

        N_pannel_width = 20 * layout_context.theme.font_size
        N_pannel_height = min(
            r.height, self._N_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._N_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height, N_pannel_width, N_pannel_height)

        E_pannel_width = 20 * layout_context.theme.font_size
        E_pannel_height = min(
            r.height, self._E_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._E_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height, E_pannel_width, E_pannel_height)

        Linear_Segments_pannel_width = 20 * layout_context.theme.font_size
        Linear_Segments_pannel_height = min(
            r.height, self._Linear_Segments_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Linear_Segments_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height + E_pannel_height,
                                                      Linear_Segments_pannel_width, Linear_Segments_pannel_height)

        Fit_Circle_pannel_width = 20 * layout_context.theme.font_size
        Fit_Circle_pannel_height = min(
            r.height, self._Fit_Circle_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Fit_Circle_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height + E_pannel_height + Linear_Segments_pannel_height,
                                                      Fit_Circle_pannel_width, Fit_Circle_pannel_height)

        Fit_Line_pannel_width = 20 * layout_context.theme.font_size
        Fit_Line_pannel_height = min(
            r.height, self._Fit_Line_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Fit_Line_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height + E_pannel_height + Linear_Segments_pannel_height + Fit_Circle_pannel_height,
                                                      Fit_Line_pannel_width, Fit_Line_pannel_height)

        Fit_BSpline_pannel_width = 20 * layout_context.theme.font_size
        Fit_BSpline_pannel_height = min(
            r.height, self._Fit_BSpline_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Fit_BSpline_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height + E_pannel_height
                                                  + Linear_Segments_pannel_height + Fit_Circle_pannel_height + Fit_Line_pannel_height,
                                                      Fit_BSpline_pannel_width, Fit_BSpline_pannel_height)

        Pick_Points_pannel_width = 20 * layout_context.theme.font_size
        Pick_Points_pannel_height = min(
            r.height, self._Pick_Points_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Pick_Points_pannel.frame = gui.Rect(r.x, r.y + Normal_Optimization_pannel_height + Load_pannel_height + N_pannel_height + E_pannel_height
                                                  + Linear_Segments_pannel_height + Fit_Circle_pannel_height
                                                  + Fit_Line_pannel_height + Fit_BSpline_pannel_height,
                                                      Pick_Points_pannel_width, Pick_Points_pannel_height)

        Segments_Key_pannel_width = 40 * layout_context.theme.font_size
        Segments_Key_pannel_height = min(
            r.height, self._Segments_Key_pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._Segments_Key_pannel.frame = gui.Rect(r.get_right() - Segments_Key_pannel_width, r.y,
                                                      Segments_Key_pannel_width, Segments_Key_pannel_height)


    def plane_threshold_check_fun(self, chk):
        if chk:
            plane_thd = float(self.plane_threshold_textedit.text_value)
            read_points_file_result_for_nor_opt = self.read_points_file_for_nor_opt()
            if read_points_file_result_for_nor_opt is not None:
                points, inst = read_points_file_result_for_nor_opt
                merged_points = fit_plane(points, inst, plane_thd)

                folder_path = os.path.join("./data_results", str(self.pcd_name))
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, str(self.pcd_name) + "_pred_normals_opt.txt")
                file_path_2 = os.path.join(self.pcd_path, str(self.pcd_name) + "_pred_normals_opt.txt")

                np.savetxt(file_path, merged_points, fmt="%.4f", delimiter=";")
                np.savetxt(file_path_2, merged_points, fmt="%.4f", delimiter=";")

    def _slider_test(self, new_val):
        if self.pcd_name is not None:
            if not self.p_switch_value:
                self.probability_val = new_val
                self.gen_edge_points(self.probability_val)
            else:
                print("Locked!")

    def p_check_fun(self, chk):
        if chk:
            self.p_switch_value = True
        else:
            self.p_switch_value = False

    def n_toggle(self, is_on):
        if self.pcd_name is not None:
            if not is_on:
                self.n_switch_edit.text_value = "------------No Normal------------"
                self._scene.scene.show_geometry(self.pcd_name + '_pred_normals', False)
            else:
                self.n_switch_edit.text_value = "-------------Normal-------------"
                self._scene.scene.show_geometry(self.pcd_name + '_pred_normals', True)
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def load_n_xyz(self):
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + '_pred_normals'):
                self._scene.scene.remove_geometry(self.pcd_name + '_pred_normals')

            points = self.pcd_points

            normals = normals_lineset(points)

            normal_mat = rendering.MaterialRecord()
            normal_mat.shader = 'unlitLine'
            normal_mat.line_width = 3

            self._scene.scene.add_geometry(self.pcd_name + '_pred_normals', normals, normal_mat)
            self._scene.scene.show_geometry(self.pcd_name + '_pred_normals', False)
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def load_vis_inst(self):
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + "_pred_Vis_I"):
                self._scene.scene.remove_geometry(self.pcd_name + "_pred_Vis_I")

            vis_inst = self.vis_inst
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vis_inst[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(vis_inst[:, 3:6]/255)

            material = rendering.MaterialRecord()
            material.point_size = 5
            material.shader = 'defaultUnlit'

            self._scene.scene.add_geometry(self.pcd_name + "_pred_Vis_I", pcd, material)
            self._scene.scene.show_geometry(self.pcd_name + "_pred_Vis_I", self.show_vis_segments)

        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def clean_segments_slider_fun(self, clean_segments_val):
        if not self.clean_segments_check_value:
            self.clean_segments_val = clean_segments_val
        else:
            print("clean_segments_slider_fun Locked!")

    def clean_segments_check_fun(self, chk):
        if chk:
            self.clean_segments_check_value = True
            read_points_file_result = self.read_points_file()
            if read_points_file_result is not None:
                points, edge, inst, vis_inst = read_points_file_result
                points_removed, edge_removed, inst_removed, vis_inst_removed = self.clean_segments(points, edge, inst, vis_inst, int(self.clean_segments_val))
                self.load_xyz(points_removed)
                self.edge_probability = edge_removed
                self.inst = inst_removed
                self.vis_inst = vis_inst_removed
                self.load_vis_inst()
        else:
            self.clean_segments_check_value = False

    def retrieval_r_slider_fun(self, retrieval_r_val):
        if not self.retrieval_r_check_value:
            self.retrieval_r_val = retrieval_r_val
        else:
            print("Locked!")

    def retrieval_r_check_fun(self, chk):
        if chk:
            self.retrieval_r_check_value = True
        else:
            self.retrieval_r_check_value = False

    def intersecting_r_slider_fun(self, intersecting_r_val):
        if not self.intersecting_r_check_value:
            self.intersecting_r_val = intersecting_r_val
        else:
            print("Locked!")

    def intersecting_r_check_fun(self, chk):
        if chk:
            self.intersecting_r_check_value = True
        else:
            self.intersecting_r_check_value = False

    def use_circle_rm_out_check_fun(self, chk):
        if chk:
            self.circle_rm_out_layout.visible = True
            self.window.set_needs_layout()
        else:
            self.circle_rm_out_layout.visible = False
            self.window.set_needs_layout()

    def use_line_rm_out_check_fun(self, chk):
        if chk:
            self.line_rm_out_layout.visible = True
            self.window.set_needs_layout()
        else:
            self.line_rm_out_layout.visible = False
            self.window.set_needs_layout()

    def use_bspline_rm_out_check_fun(self, chk):
        if chk:
            self.bspline_rm_out_layout.visible = True
            self.window.set_needs_layout()
        else:
            self.bspline_rm_out_layout.visible = False
            self.window.set_needs_layout()

    def circle_rm_out_check_fun(self, chk):
        if chk:
            self.circle_rm_out_check_value = True
        else:
            self.circle_rm_out_check_value = False

    def line_rm_out_check_fun(self, chk):
        if chk:
            self.line_rm_out_check_value = True
        else:
            self.line_rm_out_check_value = False

    def bspline_rm_out_check_fun(self, chk):
        if chk:
            self.bspline_rm_out_check_value = True
        else:
            self.bspline_rm_out_check_value = False

    def circle_rm_out_slider_fun(self, circle_rm_out_val):
        if not self.circle_rm_out_check_value:
            self.circle_rm_out_val = circle_rm_out_val
        else:
            print("circle_rm_out_slider Locked!")

    def line_rm_out_slider_fun(self, line_rm_out_val):
        if not self.line_rm_out_check_value:
            self.line_rm_out_val = line_rm_out_val
        else:
            print("line_rm_out_slider Locked!")

    def bspline_rm_out_slider_fun(self, bspline_rm_out_val):
        if not self.bspline_rm_out_check_value:
            self.bspline_rm_out_val = bspline_rm_out_val
        else:
            print("bspline_rm_out_slider Locked!")

    def circle_r_ball_check_fun(self, chk):
        if chk:
            self.circle_r_ball_check_value = True
        else:
            self.circle_r_ball_check_value = False

    def circle_n_gap_slider_fun(self, circle_n_gap_val):
        if not self.circle_n_gap_check_value:
            self.circle_n_gap_val = circle_n_gap_val
        else:
            print("circle_n_gap_slider Locked!")

    def circle_n_gap_check_fun(self, chk):
        if chk:
            self.circle_n_gap_check_value = True
        else:
            self.circle_n_gap_check_value = False

    def circle_residuals_threshold_slider_fun(self, circle_residuals_threshold_val):
        if not self.circle_residuals_threshold_check_value:
            self.circle_residuals_threshold_val = circle_residuals_threshold_val
        else:
            print("circle_residuals_threshold_slider Locked!")

    def line_residuals_threshold_slider_fun(self, line_residuals_threshold_val):
        if not self.line_residuals_threshold_check_value:
            self.line_residuals_threshold_val = line_residuals_threshold_val
        else:
            print("line_residuals_threshold_slider Locked!")

    def circle_residuals_threshold_check_fun(self, chk):
        if chk:
            self.circle_residuals_threshold_check_value = True
        else:
            self.circle_residuals_threshold_check_value = False

    def line_residuals_threshold_check_fun(self, chk):
        if chk:
            self.line_residuals_threshold_check_value = True
        else:
            self.line_residuals_threshold_check_value = False

    def seg_or_fit_combo_fun(self, val, ind):

        self.seg_or_fit_text = val

    def open_fun(self):
        if not self.do_editing_points:
            self.edit_pcd = None
            self.edit_pcd_kdtree = None
            self._picked_indicates = []
            self._picked_positions = []
            self._picked_num = 0
            self._picked_label3d = []

            seg_key = self.seg_key_textedit.text_value
            if seg_key != "" and self.seg_or_fit_text != "":
                lock_check_value, visualize_check_value, self.edit_points = self.prepare_remove_infor(seg_key, self.seg_or_fit_text)
                if lock_check_value is not None and visualize_check_value is not None and self.edit_points is not None:
                    if (not lock_check_value) and visualize_check_value:
                        self.add_visualize_edit_points(seg_key, self.edit_points)

                        self.do_editing_points = True
                        self._scene.set_on_mouse(self._mouse_event)
                        self.edit_pcd = o3d.geometry.PointCloud()
                        self.edit_pcd.points = o3d.utility.Vector3dVector(self.edit_points)
                        self.edit_pcd_kdtree = o3d.geometry.KDTreeFlann(self.edit_pcd)

                    else:
                        self.window.show_message_box('NOTE', 'Please unlock and open the corresponding visualization!')
                else:
                    self.window.show_message_box('NOTE', 'Please check the correct seg_key!')
            else:
                self.window.show_message_box('NOTE', 'Please input and select seg_key!')
        else:
            self.window.show_message_box('NOTE', 'Editable operation has been opened!')


    def close_fun(self):
        seg_key = self.seg_key_textedit.text_value
        if self.do_editing_points:
            self.do_editing_points = False
            self._scene.set_on_mouse(self._mouse_event)
            while self._picked_num > 0:
                self._scene.scene.remove_geometry('sphere' + str(self._picked_num))
                self._picked_num -= 1
                self._scene.remove_3d_label(self._picked_label3d.pop())
            if self._scene.scene.has_geometry(self.pcd_name + "_edit_points_" + str(seg_key)):
                self._scene.scene.remove_geometry(self.pcd_name + "_edit_points_" + str(seg_key))
            self._scene.force_redraw()


            self.edit_pcd = None
            self.edit_pcd_kdtree = None
            self._picked_indicates = []
            self._picked_positions = []
            self._picked_num = 0
            self._picked_label3d = []
        else:
            self.window.show_message_box('NOTE', 'Editable operation has been closed!')

    def remove_fun(self):
        unique_to_remove = list(set(self._picked_indicates))
        remove_edit_points = np.delete(self.edit_points, unique_to_remove, axis=0)
        seg_key = self.seg_key_textedit.text_value
        seg_or_fit = self.seg_or_fit_text
        if seg_or_fit == 'Seg':
            self.inter_dic[seg_key]["edited_points"] = remove_edit_points
            self.add_visualize_edited_segments(seg_key, self.inter_dic[seg_key])

        elif seg_or_fit == 'Fit':
            self.inter_dic[seg_key]["edited_fit_points"] = remove_edit_points


        self.close_fun()
        print("remove_fun")



    def invert_fun(self):
        print("invert_fun")

    def lock_all_check_fun(self, chk):
        for i, segments_key in enumerate(self.segments_key_list):
            self.seg_layouts[i].lock_check_value = chk
            self.seg_layouts[i].lock_check.checked = chk
            self.show_lock_count_fun()

    def visualize_all_segments_check_fun(self, chk):
        for i, segments_key in enumerate(self.segments_key_list):
            self.seg_layouts[i].visualize_segments_check_value = chk
            self.seg_layouts[i].visualize_segments_check.checked = chk
            seg_key = self.seg_layouts[i].key_textedit.text_value
            self.visualize_segments(seg_key, chk)

    def visualize_all_fitting_check_fun(self, chk):
        for i, segments_key in enumerate(self.segments_key_list):
            self.seg_layouts[i].visualize_fitting_check_value = chk
            self.seg_layouts[i].visualize_fitting_check.checked = chk
            seg_key = self.seg_layouts[i].key_textedit.text_value
            type = self.seg_layouts[i].type_textedit.text_value
            if type == "circle":
                self.visualize_circle_fitting(seg_key, chk)
            elif type == "line":
                self.visualize_line_fitting(seg_key, chk)
            elif type == "bspline":
                self.visualize_bspline_fitting(seg_key, chk)

    def visualize_order_points_check_fun(self, chk):
        for i, segments_key in enumerate(self.segments_key_list):
            seg_key = self.seg_layouts[i].key_textedit.text_value
            type = self.seg_layouts[i].type_textedit.text_value
            if type == "bspline":
                self.visualize_bspline_order_midpoints(seg_key, chk)

    def save_all_segments_fun(self):

        if self.inter_dic is not None:
            inter = np.empty(shape=(0, 6))
            save_seg_all_dict = {}
            for seg_key, value in self.inter_dic.items():
                seg_key_points = value["points"]
                color = value["color"].reshape(1, 3)
                points_with_color = np.hstack((seg_key_points, color.repeat(len(seg_key_points), axis=0)))
                inter = np.append(inter, points_with_color, axis=0)

                save_seg_all_dict[seg_key] = {}
                save_seg_all_dict[seg_key]["points"] = value["inter_endpoints"]
                save_seg_all_dict[seg_key]["color"] = value["color"]



            temp_key = self.segments_key_list[0]
            probability = self.inter_dic[temp_key]["probability"]
            retrieval_radius = self.inter_dic[temp_key]["retrieval_radius"]
            intersecting_radius = self.inter_dic[temp_key]["intersecting_radius"]

            folder_path = os.path.join("./data_results", str(self.pcd_name))
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius,
                                                                           intersecting_radius)
                                     + "_seg_all.txt")
            np.savetxt(file_path, inter, fmt="%.6f", delimiter=";")


            save_seg_all_dict_file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius,
                                                                           intersecting_radius)
                                     + "_seg_all.npy")
            np.save(save_seg_all_dict_file_path, save_seg_all_dict)

            self.window.show_message_box('SAVE All', 'All segments have been successfully saved!')

    def save_all_fitting_fun(self):
        if self.inter_dic is not None:
            inter = np.empty(shape=(0, 6))
            save_fit_all_dict = {}
            for i in range(len(self.segments_key_list)):
                if self.seg_layouts[i].lock_check_value:
                    seg_key = self.seg_layouts[i].key_textedit.text_value
                    seg_type = self.seg_layouts[i].type_textedit.text_value
                    fit_type = self.inter_dic[seg_key].get("type")
                    if (fit_type is not None) and (fit_type == seg_type):
                        seg_key_fit_points = self.inter_dic[seg_key]["fit_points"]
                        color = self.inter_dic[seg_key]["color"].reshape(1, 3)
                        fit_points_with_color = np.hstack((seg_key_fit_points, color.repeat(len(seg_key_fit_points), axis=0)))
                        inter = np.append(inter, fit_points_with_color, axis=0)

                        save_fit_all_dict[seg_key] = {}
                        save_fit_all_dict[seg_key]["points"] = seg_key_fit_points
                        save_fit_all_dict[seg_key]["color"] = self.inter_dic[seg_key]["color"]


            temp_key = self.segments_key_list[0]
            probability = self.inter_dic[temp_key]["probability"]
            retrieval_radius = self.inter_dic[temp_key]["retrieval_radius"]
            intersecting_radius = self.inter_dic[temp_key]["intersecting_radius"]

            folder_path = os.path.join("./data_results", str(self.pcd_name))
            os.makedirs(folder_path, exist_ok=True)

            lock_count_text = self.lock_count_textedit.text_value
            sanitized_lock_count_text = "".join(c if c.isalnum() or c in ('_', '-') else 'of' if c == '/' else '_' for c in lock_count_text)
            file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius,
                                                                           intersecting_radius)
                                     + "_fit_" + sanitized_lock_count_text + ".txt")
            np.savetxt(file_path, inter, fmt="%.6f", delimiter=";")

            save_fit_all_dict_file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius,
                                                                           intersecting_radius)
                                     + "_fit_" + sanitized_lock_count_text + ".npy")
            np.save(save_fit_all_dict_file_path, save_fit_all_dict)


            data_list = []  # List to store extracted data
            for i in range(len(self.segments_key_list)):
                if self.seg_layouts[i].lock_check_value:
                    seg_key = self.seg_layouts[i].key_textedit.text_value
                    seg_type = self.seg_layouts[i].type_textedit.text_value
                    fit_type = self.inter_dic.get(seg_key, {}).get("type")
                    if fit_type is not None and fit_type == seg_type and fit_type == "line":
                        # Extract data from the nested dictionary
                        m0 = self.inter_dic[seg_key].get("m0", None)
                        m1 = self.inter_dic[seg_key].get("m1", None)
                        fit_points = self.inter_dic[seg_key].get("fit_points", None)
                        # Check if all required keys are present
                        if m0 is not None and m1 is not None and fit_points is not None:
                            # Append the data to the list along with seg_key
                            seg_key_with_pcd_name = f"{self.pcd_name}_{seg_key}"
                            data_list.append((seg_key_with_pcd_name, m0, m1, fit_points))

            if data_list:
                dtype = [('seg_key_with_pcd_name', 'U100'), ('m0', 'float64', (3,)), ('m1', 'float64', (3,)),
                         ('fit_points', object)]
                data_list_inter = np.array(data_list, dtype=dtype)
                data_list_file_path = os.path.join(folder_path, str(self.pcd_name)
                                         + "_line_information_" + sanitized_lock_count_text + ".npy")
                np.save(data_list_file_path, data_list_inter)


            self.window.show_message_box('SAVE Locked', 'All locked fitting (' + str(self.lock_count_textedit.text_value) + ') have been successfully saved!')

    def read_points_file(self):
        if self.pcd_name is not None:
            if self.plane_threshold_check.checked:
                points = np.loadtxt(self.pcd_path + "/" + self.pcd_name + "_pred_normals_opt.txt", delimiter=";")
            else:
                points = np.loadtxt(self.pcd_path + "/" + self.pcd_name + "_pred_normals.txt", delimiter=";")

            file_path = self.pcd_path + "/" + self.pcd_name + "_pred_edge.txt"
            df = pd.read_csv(file_path, delimiter=';', header=None)
            data = df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
            edge_probability = np.array(data)


            vis_inst = np.loadtxt(self.pcd_path + "/" + self.pcd_name + "_pred_Vis_I.txt", delimiter=";")
            if os.path.isfile(self.pcd_path + "/" + self.pcd_name + '_pred_inst.npy'):
                inst = np.load(self.pcd_path + "/" + self.pcd_name + '_pred_inst.npy').astype(int)  # npy
            elif os.path.isfile(self.pcd_path + "/" + self.pcd_name + '_pred_inst.txt'):
                inst = np.loadtxt(self.pcd_path + "/" + self.pcd_name + '_pred_inst.txt', dtype=int) # txt
            return points, edge_probability, inst, vis_inst
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def read_points_file_for_nor_opt(self):
        if self.pcd_name is not None:
            points = np.loadtxt(self.pcd_path + "/" + self.pcd_name + "_pred_normals.txt", delimiter=";")

            if os.path.isfile(self.pcd_path + "/" + self.pcd_name + '_pred_inst.npy'):
                inst = np.load(self.pcd_path + "/" + self.pcd_name + '_pred_inst.npy').astype(int)  # npy
            elif os.path.isfile(self.pcd_path + "/" + self.pcd_name + '_pred_inst.txt'):
                inst = np.loadtxt(self.pcd_path + "/" + self.pcd_name + '_pred_inst.txt', dtype=int) # txt
            return points, inst
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def clean_segments(self, points, edge_probability, inst_num, vis_inst, clean_thd):
        out_index = []
        new_vis = np.unique(inst_num, axis=0)
        for color in new_vis:
            index = np.where(inst_num == color)
            length = len(index[0])
            if length < clean_thd:
                print(color, "-->clean length: ", length)
                out_index = np.append(out_index, index)
        if len(out_index):
            out_index = out_index.astype(int)

        points_removed = np.delete(points, out_index, axis=0)
        edge_probability_removed = np.delete(edge_probability, out_index, axis=0)
        inst_num_removed = np.delete(inst_num, out_index, axis=0)
        vis_inst_removed = np.delete(vis_inst, out_index, axis=0)
        return points_removed, edge_probability_removed, inst_num_removed, vis_inst_removed

    def linear_segments_start_fun(self):
        if self.p_switch_value and self.retrieval_r_check_value and self.intersecting_r_check_value:
            self.inter_dic = self.gen_linear_segments(self.retrieval_r_val, self.intersecting_r_val, self.probability_val)
            if self.inter_dic is not None:
                self.segments_key_list = list(self.inter_dic.keys())

                for i, segments_key in enumerate(self.segments_key_list):
                    self.seg_layouts[i].seg_layout.visible = True
                    self.seg_layouts[i].key_textedit.text_value = f'{segments_key}'
                    color = self.inter_dic[segments_key]["color"]/255

                    self.seg_layouts[i].save_segment_button.background_color = gui.Color(color[0], color[1], color[2])

                for j in range(len(self.segments_key_list), 100):
                    self.seg_layouts[j].seg_layout.visible = False
                self.window.set_needs_layout()
                self.show_lock_count_fun()
                # remove linear segments
                self.remove_linear_segments()
                # visualize segments
                for i, segments_key in enumerate(self.segments_key_list):
                    self.add_visualize_segments(segments_key, self.inter_dic[segments_key], self.seg_layouts[i].visualize_segments_check_value)
                # self.visualize_linear_segments(self.inter_dic)
        else:
            self.window.show_message_box('NOTE', 'Make sure that Probability, Retrieval Radius, and Intersecting Radius have ALL been checked!')

    def circle_fit_start_fun(self):
        if self.inter_dic is not None:
            if self.circle_r_ball_check_value and self.circle_n_gap_check_value and self.circle_residuals_threshold_check_value:
                for i in range(len(self.segments_key_list)):
                    if not self.seg_layouts[i].lock_check_value:
                        seg_key = self.seg_layouts[i].key_textedit.text_value
                        value_points = self.inter_dic[seg_key]["points"]
                        self.gen_circle_fit(seg_key, value_points,
                                            self.circle_rm_out_layout.visible, int(self.circle_r_ball_textedit.text_value),
                                            int(self.circle_n_gap_val), self.circle_residuals_threshold_val)
                # visualize
                for i, segments_key in enumerate(self.segments_key_list):
                    if not self.seg_layouts[i].lock_check_value:
                        c_res = self.inter_dic[segments_key].get("circle_residuals")
                        if c_res is not None:
                            self.seg_layouts[i].circle_residual_textedit.text_value = c_res if isinstance(c_res, str) else "{:.5f}".format(c_res)

                        fit_type = self.inter_dic[segments_key].get("type")
                        if (fit_type is not None) and (fit_type == "circle"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.add_visualize_circle_fitting(segments_key, self.inter_dic[segments_key], self.seg_layouts[i].visualize_fitting_check_value)
                        if (fit_type is not None) and (fit_type == "error"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)
                        if (fit_type is not None) and (fit_type == ""):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)
            else:
                self.window.show_message_box('NOTE', 'Make sure that Ball Radius Search, Gap in the Arc, and Circle Residuals Threshold have ALL been checked!')

    def line_fit_start_fun(self):
        if self.inter_dic is not None:
            if self.line_residuals_threshold_check_value:
                for i in range(len(self.segments_key_list)):
                    if not self.seg_layouts[i].lock_check_value:
                        seg_key = self.seg_layouts[i].key_textedit.text_value
                        value_points = self.inter_dic[seg_key]["points"]
                        self.gen_line_fit(seg_key, value_points,
                                            self.line_rm_out_layout.visible, self.line_residuals_threshold_val)

                # visualize
                for i, segments_key in enumerate(self.segments_key_list):
                    if not self.seg_layouts[i].lock_check_value:
                        l_res = self.inter_dic[segments_key].get("line_residuals")
                        if l_res is not None:
                            self.seg_layouts[i].line_residual_textedit.text_value = "{:.5f}".format(l_res)

                        fit_type = self.inter_dic[segments_key].get("type")
                        if (fit_type is not None) and (fit_type == "line"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.add_visualize_line_fitting(segments_key, self.inter_dic[segments_key], self.seg_layouts[i].visualize_fitting_check_value)
                        if (fit_type is not None) and (fit_type == "error"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)
                        if (fit_type is not None) and (fit_type == ""):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)

            else:
                self.window.show_message_box('NOTE', 'Make sure that Line Residuals Threshold have been checked!')

    def order_points_start_fun(self):
        if self.p_switch_value and self.retrieval_r_check_value and self.intersecting_r_check_value:
            self.midpoints_dic = self.gen_linear_segments_midpoints(self.retrieval_r_val, self.intersecting_r_val)
            if self.midpoints_dic is not None:
                for i in range(len(self.segments_key_list)):
                    if not self.seg_layouts[i].lock_check_value:
                        seg_key = self.seg_layouts[i].key_textedit.text_value
                        value_points = self.midpoints_dic[seg_key]
                        self.gen_order_points(seg_key, value_points, self.bspline_rm_out_layout.visible)

                # visualize
                for i, segments_key in enumerate(self.segments_key_list):
                    if not self.seg_layouts[i].lock_check_value:
                        fit_type = self.inter_dic[segments_key].get("type")
                        if (fit_type is not None) and (fit_type == "bspline"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.add_visualize_bspline_order_midpoints(segments_key, self.inter_dic[segments_key], self.visualize_order_points_check_value)
                        if (fit_type is not None) and (fit_type == "error"):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)
                        if (fit_type is not None) and (fit_type == ""):
                            self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                            self.clean_visualize_fitting(segments_key)
        else:
            self.window.show_message_box('NOTE',
                                             'Make sure that Probability, Retrieval Radius, and Intersecting Radius have ALL been checked!')

    def bspline_fit_start_fun(self):
        if self.midpoints_dic is not None:
            order_midpoints = {}
            for i in range(len(self.segments_key_list)):
                if not self.seg_layouts[i].lock_check_value:
                    seg_key = self.seg_layouts[i].key_textedit.text_value
                    fit_type = self.seg_layouts[i].type_textedit.text_value
                    bspline_order_midpoints = self.inter_dic[seg_key].get("bspline_order_midpoints")
                    if (fit_type == "bspline") and (bspline_order_midpoints is not None):
                        order_midpoints[seg_key] = bspline_order_midpoints

            self.gen_bspline_fit(order_midpoints)

            # visualize
            for i, segments_key in enumerate(self.segments_key_list):
                if not self.seg_layouts[i].lock_check_value:
                    fit_type = self.inter_dic[segments_key].get("type")
                    if (fit_type is not None) and (fit_type == "bspline"):
                        self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                        self.add_visualize_bspline_fitting(segments_key, self.inter_dic[segments_key], self.seg_layouts[i].visualize_fitting_check_value)
                    if (fit_type is not None) and (fit_type == ""):
                        self.seg_layouts[i].type_textedit.text_value = f'{fit_type}'
                        self.clean_visualize_fitting(segments_key)
        else:
            self.window.show_message_box('NOTE',
                                             'Please first click on Start Sorting Points!')

    def show_lock_count_fun(self):
        if self.inter_dic is not None:
            all_count = len(self.segments_key_list)
            true_count = sum(layout_item.lock_check_value for layout_item in self.seg_layouts[:all_count])
            self.lock_count_textedit.text_value = f"{int(true_count)}/{int(all_count)}"

    def gen_edge_points(self, P):
        print("P", P)
        if self.pcd_name is not None:
            if self._scene.scene.has_geometry(self.pcd_name + "_pred_edge"):
                self._scene.scene.remove_geometry(self.pcd_name + "_pred_edge")

            edge_points = np.where(self.edge_probability >= P)
            self.e_list = edge_points[0]

            points = self.pcd_points
            edge_p = points[self.e_list]
            print("edge_p.shape", edge_p.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(edge_p[:, 0:3])
            pcd.paint_uniform_color([1, 0.706, 0])

            material = rendering.MaterialRecord()
            material.point_size = 7
            material.shader = 'defaultUnlit'
            self._scene.scene.add_geometry(self.pcd_name + "_pred_edge", pcd, material)
            self._scene.scene.show_geometry(self.pcd_name + "_pred_edge", self.show_edge_points)
            # bounds = pcd.get_axis_aligned_bounding_box()
            # self._scene.setup_camera(60, bounds, bounds.get_center())
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')

    def gen_linear_segments(self, retrieval_radius, R, P):
        if self.pcd_name is not None:

            linesample = 10
            inter_dict = {}

            gt_points = self.pcd_points
            e_list = self.e_list
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_points[:, 0:3])
            pcd.normals = o3d.utility.Vector3dVector(gt_points[:, 3:6])

            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            for e in e_list:
                [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[e], retrieval_radius) # retrieval_radius
                neighborindex_list = idx[1:]
                C1 = np.array(gt_points[e][:3])
                N1 = np.array(gt_points[e][3:])

                for neighborindex in neighborindex_list:
                    line_points = np.empty(shape=(0, 3))
                    if(self.inst[e] != self.inst[neighborindex]):
                        C2 = np.array(gt_points[neighborindex][:3])
                        N2 = np.array(gt_points[neighborindex][3:])
                        inter_p1, inter_p2 = get_intersection(C1, N1, C2, N2, R)
                        # print("--------------------------------------------")
                        if (True not in np.isnan(inter_p1)) and (True not in np.isnan(inter_p2)): # remove NAN
                            X = np.linspace(start=inter_p1[0], stop=inter_p2[0], num=linesample).reshape(linesample, 1)
                            Y = np.linspace(start=inter_p1[1], stop=inter_p2[1], num=linesample).reshape(linesample, 1)
                            Z = np.linspace(start=inter_p1[2], stop=inter_p2[2], num=linesample).reshape(linesample, 1)
                            line_points = np.concatenate((X, Y, Z), axis=1)

                            inter_p1_p2 = np.vstack((inter_p1, inter_p2))

                        line_points = line_points[[not np.all(line_points[i] == 0) for i in range(line_points.shape[0])], :]  # remove 0
                        if line_points.shape[0] != 0:
                            inst_key = str(self.inst[e]) + '-' + str(self.inst[neighborindex])
                            if inst_key in inter_dict.keys():
                                inter_dict[inst_key]["points"] = np.append(inter_dict[inst_key]["points"], line_points, axis=0)
                                inter_dict[inst_key]["inter_endpoints"] = np.append(inter_dict[inst_key]["inter_endpoints"], inter_p1_p2, axis=0)
                            else:
                                inter_dict[inst_key] = {}
                                inter_dict[inst_key]["points"] = line_points
                                inter_dict[inst_key]["probability"] = P
                                inter_dict[inst_key]["retrieval_radius"] = retrieval_radius
                                inter_dict[inst_key]["intersecting_radius"] = R
                                inter_dict[inst_key]["inter_endpoints"] = inter_p1_p2
            inter_dic = combine_key(inter_dict)
            return inter_dic
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')
            return None

    def gen_linear_segments_midpoints(self, retrieval_radius, R):
        if self.pcd_name is not None:

            midpoints_dict = {}

            gt_points = self.pcd_points
            e_list = self.e_list
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_points[:, 0:3])
            pcd.normals = o3d.utility.Vector3dVector(gt_points[:, 3:6])

            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            for e in e_list:
                [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[e], retrieval_radius) # retrieval_radius
                neighborindex_list = idx[1:]
                C1 = np.array(gt_points[e][:3])
                N1 = np.array(gt_points[e][3:])

                for neighborindex in neighborindex_list:
                    line_points = np.empty(shape=(0, 3))
                    if(self.inst[e] != self.inst[neighborindex]):
                        C2 = np.array(gt_points[neighborindex][:3])
                        N2 = np.array(gt_points[neighborindex][3:])
                        inter_p1, inter_p2 = get_intersection(C1, N1, C2, N2, R)
                        # print("--------------------------------------------")
                        if (True not in np.isnan(inter_p1)) and (True not in np.isnan(inter_p2)): # remove NAN
                            line_points = (inter_p1 + inter_p2) / 2
                            line_points = np.reshape(line_points, (1, 3))
                        line_points = line_points[[not np.all(line_points[i] == 0) for i in range(line_points.shape[0])], :]  # remove 0
                        if line_points.shape[0] != 0:
                            inst_key = str(self.inst[e]) + '-' + str(self.inst[neighborindex])
                            if inst_key in midpoints_dict.keys():
                                midpoints_dict[inst_key] = np.append(midpoints_dict[inst_key], line_points, axis=0)
                            else:
                                midpoints_dict[inst_key] = line_points

            midpoints_dict = midpoints_combine_key(midpoints_dict)
            return midpoints_dict
        else:
            self.window.show_message_box('ERROR', 'Please open the points_pred_normals file first!')
            return None

    def remove_linear_segments(self):
        for segments_key_temp in self.segments_key_list_temp:
            if self._scene.scene.has_geometry(self.pcd_name + "_segment_" + str(segments_key_temp)):
                self._scene.scene.remove_geometry(self.pcd_name + "_segment_" + str(segments_key_temp))
            if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(segments_key_temp)):
                self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(segments_key_temp))
            if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(segments_key_temp)):
                self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(segments_key_temp))
            if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(segments_key_temp)):
                self._scene.scene.remove_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(segments_key_temp))
            if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(segments_key_temp)):
                self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(segments_key_temp))

        self.segments_key_list_temp = self.segments_key_list
        [self._scene.remove_3d_label(value_list[0]) for value_list in self.segments_key_label_dic.values() if value_list]
        self.segments_key_label_dic = {}

    def clean_visualize_fitting(self, seg_key):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key))

    def add_visualize_segments(self, seg_key, seg_value, vis_seg_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_segment_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_segment_" + str(seg_key))

        # visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["points"])
        color = adjust_brightness(seg_value["color"], 0.6)
        pcd.paint_uniform_color(color)

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_segment_" + str(seg_key), pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_segment_" + str(seg_key), vis_seg_flag)
        seg_key_label = self._scene.add_3d_label(seg_value["points"][0], str(seg_key))
        self.segments_key_label_dic[seg_key] = [seg_key_label, seg_value["points"][0]]
        if not vis_seg_flag:
            self._scene.remove_3d_label(self.segments_key_label_dic[seg_key][0])

    def add_visualize_circle_fitting(self, seg_key, seg_value, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["fit_points"])
        pcd.paint_uniform_color(seg_value["color"]/255)

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key), pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key), vis_fit_flag)

    def add_visualize_line_fitting(self, seg_key, seg_value, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["fit_points"])
        pcd.paint_uniform_color(seg_value["color"]/255)

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_line_fitting_" + str(seg_key), pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_line_fitting_" + str(seg_key), vis_fit_flag)

    def add_visualize_bspline_order_midpoints(self, seg_key, seg_value, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["bspline_order_midpoints"])

        # get colormap
        cmap = plt.get_cmap("gist_rainbow")
        n = seg_value["bspline_order_midpoints"].shape[0]
        norm = Normalize(vmin=0, vmax=n - 1)
        colors = cmap(norm(range(n)))
        # colors
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key), pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key), vis_fit_flag)

    def add_visualize_bspline_fitting(self, seg_key, seg_value, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key))
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_line_fitting_" + str(seg_key))

        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["fit_points"])
        pcd.paint_uniform_color(seg_value["color"]/255)

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key), pcd, material)
        self._scene.scene.show_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key), vis_fit_flag)

    def add_visualize_edit_points(self, seg_key, points):
        if self._scene.scene.has_geometry(self.pcd_name + "_edit_points_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_edit_points_" + str(seg_key))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 0, 1])

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'
        self._scene.scene.add_geometry(self.pcd_name + "_edit_points_" + str(seg_key), pcd, material)

    def add_visualize_edited_segments(self, seg_key, seg_value):
        if self._scene.scene.has_geometry(self.pcd_name + "_segment_" + str(seg_key)):
            self._scene.scene.remove_geometry(self.pcd_name + "_segment_" + str(seg_key))


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg_value["edited_points"])
        color = adjust_brightness(seg_value["color"], 0.6)
        pcd.paint_uniform_color(color)

        material = rendering.MaterialRecord()
        material.point_size = 7
        material.shader = 'defaultUnlit'

        self._scene.scene.add_geometry(self.pcd_name + "_segment_" + str(seg_key), pcd, material)
        if self.segments_key_label_dic[seg_key]:
            self._scene.remove_3d_label(self.segments_key_label_dic[seg_key][0])
            seg_key_label = self._scene.add_3d_label(self.segments_key_label_dic[seg_key][1], str(seg_key))
            self.segments_key_label_dic[seg_key][0] = seg_key_label # refresh label dictionary
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Linear Segments!')

    def visualize_segments(self, seg_key, vis_seg_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_segment_" + str(seg_key)):
            self._scene.scene.show_geometry(self.pcd_name + "_segment_" + str(seg_key), vis_seg_flag)
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Linear Segments!')

        if self.segments_key_label_dic[seg_key]:
            if vis_seg_flag:
                self._scene.remove_3d_label(self.segments_key_label_dic[seg_key][0])
                seg_key_label = self._scene.add_3d_label(self.segments_key_label_dic[seg_key][1], str(seg_key))
                self.segments_key_label_dic[seg_key][0] = seg_key_label
            else:
                self._scene.remove_3d_label(self.segments_key_label_dic[seg_key][0])
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Linear Segments!')

    def visualize_circle_fitting(self, seg_key, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key)):
            self._scene.scene.show_geometry(self.pcd_name + "_circle_fitting_" + str(seg_key), vis_fit_flag)
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Circles!')

    def visualize_line_fitting(self, seg_key, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_line_fitting_" + str(seg_key)):
            self._scene.scene.show_geometry(self.pcd_name + "_line_fitting_" + str(seg_key), vis_fit_flag)
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Lines!')

    def visualize_bspline_order_midpoints(self, seg_key, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key)):
            self._scene.scene.show_geometry(self.pcd_name + "_bspline_order_midpoints_" + str(seg_key), vis_fit_flag)
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Sorting Points!')

    def visualize_bspline_fitting(self, seg_key, vis_fit_flag):
        if self._scene.scene.has_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key)):
            self._scene.scene.show_geometry(self.pcd_name + "_bspline_fitting_" + str(seg_key), vis_fit_flag)
        else:
            self.window.show_message_box('NOTE',
                                         'Please first click on Start Generating Bspline!')

    def save_segments(self, seg_key):
        if self.inter_dic is not None:
            save_seg_dict = {}

            seg_key_points = self.inter_dic[seg_key]["points"]
            probability = self.inter_dic[seg_key]["probability"]
            retrieval_radius = self.inter_dic[seg_key]["retrieval_radius"]
            intersecting_radius = self.inter_dic[seg_key]["intersecting_radius"]
            color = self.inter_dic[seg_key]["color"].reshape(1, 3)
            points_with_color = np.hstack((seg_key_points, color.repeat(len(seg_key_points), axis=0)))

            save_seg_dict[seg_key] = {}
            save_seg_dict[seg_key]["points"] = self.inter_dic[seg_key]["inter_endpoints"]
            save_seg_dict[seg_key]["color"] = self.inter_dic[seg_key]["color"]

            folder_path = os.path.join("./data_results", str(self.pcd_name))
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius, intersecting_radius)
                                     + "_seg" + str(seg_key) + ".txt")
            np.savetxt(file_path, points_with_color, fmt="%.6f", delimiter=";")

            save_seg_dict_file_path = os.path.join(folder_path, str(self.pcd_name)
                                     + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius, intersecting_radius)
                                     + "_seg" + str(seg_key) + ".npy")
            np.save(save_seg_dict_file_path, save_seg_dict)


            self.window.show_message_box('SAVE', 'The seg ' + str(seg_key) + ' has been successfully saved!')

    def save_fittings(self, seg_key, seg_type):
        if self.inter_dic is not None:
            save_fit_dict = {}

            fit_type = self.inter_dic[seg_key].get("type")
            if (fit_type is not None) and (fit_type == seg_type):
                seg_key_fit_points = self.inter_dic[seg_key]["fit_points"]
                probability = self.inter_dic[seg_key]["probability"]
                retrieval_radius = self.inter_dic[seg_key]["retrieval_radius"]
                intersecting_radius = self.inter_dic[seg_key]["intersecting_radius"]
                color = self.inter_dic[seg_key]["color"].reshape(1, 3)
                fit_points_with_color = np.hstack((seg_key_fit_points, color.repeat(len(seg_key_fit_points), axis=0)))

                save_fit_dict[seg_key] = {}
                save_fit_dict[seg_key]["points"] = seg_key_fit_points
                save_fit_dict[seg_key]["color"] = self.inter_dic[seg_key]["color"]


                folder_path = os.path.join("./data_results", str(self.pcd_name))
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, str(self.pcd_name)
                                         + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius, intersecting_radius)
                                         + "_fit" + str(seg_key) + "_" + str(fit_type) + ".txt")
                np.savetxt(file_path, fit_points_with_color, fmt="%.6f", delimiter=";")

                save_fit_dict_file_path = os.path.join(folder_path, str(self.pcd_name)
                                         + "_p{:.2f}_rr{:.3f}_ir{:.3f}".format(probability, retrieval_radius, intersecting_radius)
                                         + "_fit" + str(seg_key) + "_" + str(fit_type) + ".npy")
                np.save(save_fit_dict_file_path, save_fit_dict)

                self.window.show_message_box('SAVE', 'The fitting ' + str(seg_key) + "_" + str(fit_type)+ ' has been successfully saved!')

    def gen_circle_fit(self, seg_key, value, circle_rm_out_flag, r_ball_multiple, n_gap, residuals_thd):

        self.inter_dic[seg_key]["use_circle_rm_out"] = circle_rm_out_flag
        self.inter_dic[seg_key]["r_ball"] = r_ball_multiple
        self.inter_dic[seg_key]["n_gap"] = n_gap
        self.inter_dic[seg_key]["circle_residuals_thd"] = residuals_thd
        if circle_rm_out_flag:
            if self.circle_rm_out_check_value:
                value = remove_outlier(value, int(self.circle_rm_out_val))
                self.inter_dic[seg_key]["circle_rm_out_val"] = int(self.circle_rm_out_val)
                self.inter_dic[seg_key]["circle_rm_out_array"] = value
            else:
                self.window.show_message_box('NOTE',
                                             'Make sure that Remove_Outliers have been checked!')
        else:
            self.inter_dic[seg_key]["circle_rm_out_val"] = None
            self.inter_dic[seg_key]["circle_rm_out_array"] = None

        circle, circle_center, radius, c_residuals, normal, u = circle_segmentation(np.array(value))
        print("seg_key", seg_key)
        if (c_residuals.size == 0):
            self.inter_dic[seg_key]["circle_residuals"] = "None"

            self.inter_dic[seg_key]["type"] = ""
            self.inter_dic[seg_key]["fit_points"] = np.empty((0, 3))
            pass
        elif (c_residuals[0] > residuals_thd):
            self.inter_dic[seg_key]["circle_residuals"] = c_residuals[0]
            print("c_residuals -> non-circle", c_residuals)

            self.inter_dic[seg_key]["type"] = ""
            self.inter_dic[seg_key]["fit_points"] = np.empty((0, 3))
        else:
            self.inter_dic[seg_key]["circle_residuals"] = c_residuals[0]
            self.inter_dic[seg_key]["circle_center"] = circle_center
            self.inter_dic[seg_key]["circle_radius"] = radius
            self.inter_dic[seg_key]["circle_normal"] = normal

            p2p_dis = get_r_ball(circle)
            t = find_arc(value, circle, p2p_dis, r_ball_multiple, n_gap)
            t_rad = deg2rad(t)
            fitcircle = sample_circle(t_rad, circle_center, radius, normal, u)
            if len(fitcircle) == 0:
                self.inter_dic[seg_key]["type"] = "error"
            else:
                self.inter_dic[seg_key]["type"] = "circle"
            self.inter_dic[seg_key]["fit_points"] = fitcircle
            self.inter_dic[seg_key]["circle_arc"] = t

    def gen_line_fit(self, seg_key, value, line_rm_out_flag, residuals_thd):

        self.inter_dic[seg_key]["use_line_rm_out"] = line_rm_out_flag
        self.inter_dic[seg_key]["line_residuals_thd"] = residuals_thd
        if line_rm_out_flag:
            if self.line_rm_out_check_value:
                value = remove_outlier(value, int(self.line_rm_out_val))
                self.inter_dic[seg_key]["line_rm_out_val"] = int(self.line_rm_out_val)
                self.inter_dic[seg_key]["line_rm_out_array"] = value
            else:
                self.window.show_message_box('NOTE',
                                             'Make sure that Remove_Outliers have been checked!')
        else:
            self.inter_dic[seg_key]["line_rm_out_val"] = None
            self.inter_dic[seg_key]["line_rm_out_array"] = None

        l_residuals, z0, z1, line_points = fit_xyz_yzx_xzy(np.array(value))

        self.inter_dic[seg_key]["line_residuals"] = l_residuals
        print("fit line seg_key", seg_key, l_residuals)
        if l_residuals > residuals_thd:
            print("none l_residuals ", l_residuals)
            self.inter_dic[seg_key]["type"] = ""
            self.inter_dic[seg_key]["fit_points"] = np.empty((0, 3))
        else:
            if len(line_points) == 0:
                self.inter_dic[seg_key]["type"] = "error"
            else:
                self.inter_dic[seg_key]["type"] = "line"
            self.inter_dic[seg_key]["fit_points"] = line_points
            self.inter_dic[seg_key]["m0"] = z0
            self.inter_dic[seg_key]["m1"] = z1

    def gen_order_points(self, seg_key, value_unorder, bspline_rm_out_flag):

        self.inter_dic[seg_key]["bspline_midpoints"] = value_unorder
        self.inter_dic[seg_key]["use_bspline_rm_out"] = bspline_rm_out_flag
        if bspline_rm_out_flag:
            if self.bspline_rm_out_check_value:
                value_unorder = remove_outlier(value_unorder, int(self.bspline_rm_out_val))
                self.inter_dic[seg_key]["bspline_rm_out_val"] = int(self.bspline_rm_out_val)
                self.inter_dic[seg_key]["bspline_rm_out_array"] = value_unorder
            else:
                self.window.show_message_box('NOTE',
                                             'Make sure that Remove_Outliers have been checked!')
        else:
            self.inter_dic[seg_key]["bspline_rm_out_val"] = None
            self.inter_dic[seg_key]["bspline_rm_out_array"] = None


        ind1, ind2 = find_glob_max(value_unorder)
        print(seg_key, "----maxdis----", ind1, ind2)
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
        print("value_order_flag.shape", value_order_flag.shape)
        order_points = value_order_flag[:, 0:3]

        if len(order_points) == 0:
            self.inter_dic[seg_key]["type"] = "error"
        else:
            self.inter_dic[seg_key]["type"] = "bspline"
        self.inter_dic[seg_key]["bspline_order_midpoints"] = order_points

    def gen_bspline_fit(self, dic_order):
        folder_path = os.path.join("./data_results", str(self.pcd_name))
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, str(self.pcd_name) + "_bspline.txt")
        with open(filename, 'w') as f:
            print("keys(): ", len(dic_order.keys()))
            f.write(str(len(dic_order.keys())) + "\n")
            key_record = []
            for key1, value1 in dic_order.items():
                key_record.append(key1)
                f.write(str(value1.shape[0]) + "\n")
                print("value.shape[0]: ", value1.shape[0])
                for row in range(len(value1)):
                    for col in range(3):
                        f.write("%.6f%c" % (value1[row][col], " \n"[col == 2]))

        # using the engine to fit the B-spline
        self.engine.fitting3D(filename, nargout=0)
        inter_pixel_list = self.engine.showbezier_3d(filename)
        all_points = np.loadtxt(filename[:-4] + "_res.txt", delimiter=";")

        flat_inter_pixel_list = np.concatenate(inter_pixel_list)
        cumulative_sum = np.concatenate(([0], np.cumsum(flat_inter_pixel_list)))
        for i, key in enumerate(key_record):
            start_index = int(cumulative_sum[i])
            end_index = int(cumulative_sum[i + 1]) if i < len(cumulative_sum) - 1 else None

            coordinates = all_points[start_index:end_index]

            self.inter_dic[key]["fit_points"] = coordinates

    def prepare_remove_infor(self, seg_key, seg_or_fit):
        try:
            seg_layout_index = self.segments_key_list.index(seg_key)
            lock_check_value = self.seg_layouts[seg_layout_index].lock_check_value
            if seg_or_fit == 'Seg':
                visualize_check_value = self.seg_layouts[seg_layout_index].visualize_segments_check_value
                edit_points = self.inter_dic[seg_key].get("points", None)
            elif seg_or_fit == 'Fit':
                visualize_check_value = self.seg_layouts[seg_layout_index].visualize_fitting_check_value
                edit_points = self.inter_dic[seg_key].get("fit_points", None)
        except ValueError:
            return None, None, None
        return lock_check_value, visualize_check_value, edit_points

    def _mouse_event(self, event):
        if self.do_editing_points == True:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL):

                def depth_callback(depth_image):
                    x = event.x - self._scene.frame.x
                    y = event.y - self._scene.frame.y
                    depth = np.asarray(depth_image)[y, x]
                    if depth == 1.0:
                        return
                    else:
                        world_coord = self._scene.scene.camera.unproject(
                            x, y, depth, self._scene.frame.width, self._scene.frame.height)
                        idx = self._calc_prefer_indicate(world_coord)
                        picked_point = np.asarray(self.edit_pcd.points)[idx]
                        self._picked_num += 1
                        self._picked_indicates.append(idx)
                        self._picked_positions.append(picked_point)

                        print(
                            f"Picked point #{idx} at ({picked_point[0]}, {picked_point[1]}, {picked_point[2]})")

                        def draw_point():
                            label3d = self._scene.add_3d_label(picked_point, "#"+str(self._picked_num))
                            self._picked_label3d.append(label3d)

                            sphere = o3d.geometry.TriangleMesh.create_sphere(0.001)
                            sphere.paint_uniform_color([1, 0, 0])
                            sphere.translate(picked_point)
                            material = rendering.MaterialRecord()
                            material.shader = 'defaultUnlit'

                            self._scene.scene.add_geometry("sphere"+str(self._picked_num), sphere, material)
                            self._scene.force_redraw()

                        gui.Application.instance.post_to_main_thread(
                            self.window, draw_point)

                self._scene.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.HANDLED
            elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT) and event.is_modifier_down(gui.KeyModifier.CTRL):
                if self._picked_num > 0:
                    idx = self._picked_indicates.pop()
                    point = self._picked_positions.pop()

                    print(
                        f"Undo pick: #{idx} at ({point[0]}, {point[1]}, {point[2]})")

                    self._scene.scene.remove_geometry('sphere'+str(self._picked_num))
                    self._picked_num -= 1
                    self._scene.remove_3d_label(self._picked_label3d.pop())
                    self._scene.force_redraw()
                else:
                    print('Undo nothing!')
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        else:
            return gui.Widget.EventCallbackResult.IGNORED
    def _calc_prefer_indicate(self, point):
        [_, idx, _] = self.edit_pcd_kdtree.search_knn_vector_3d(point, 1)
        return idx[0]


    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":

    app = App()
    app.run()

