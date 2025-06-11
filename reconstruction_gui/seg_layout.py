
import open3d.visualization.gui as gui

class SegLayout:
    lock_check_value = False
    visualize_segments_check_value = True
    visualize_fitting_check_value = True

    def __init__(self, my_app):

        self.my_app = my_app
        # self.seg_layout = gui.Vert()
        self.lock_check = gui.Checkbox("")
        self.lock_check.checked = self.lock_check_value
        self.lock_check.set_on_checked(self.lock_check_fun)
        self.key_textedit = gui.TextEdit()
        self.visualize_segments_check = gui.Checkbox("")
        self.visualize_segments_check.checked = self.visualize_segments_check_value
        self.visualize_segments_check.set_on_checked(self.visualize_segments_check_fun)
        self.visualize_fitting_check = gui.Checkbox("")
        self.visualize_fitting_check.checked = self.visualize_fitting_check_value
        self.visualize_fitting_check.set_on_checked(self.visualize_fitting_check_fun)
        self.type_textedit = gui.TextEdit()

        self.circle_residual_textedit = gui.TextEdit()
        self.line_residual_textedit = gui.TextEdit()
        self.save_segment_button = gui.Button("Save")
        self.save_segment_button.vertical_padding_em = 0.06
        self.save_segment_button.set_on_clicked(self.save_segment_fun)
        self.save_fitting_button = gui.Button("Save")
        self.save_fitting_button.vertical_padding_em = 0.06
        self.save_fitting_button.set_on_clicked(self.save_fitting_fun)
        self.reload_parameters_button = gui.Button("Reload")
        self.reload_parameters_button.vertical_padding_em = 0.06
        self.reload_parameters_button.set_on_clicked(self.reload_parameters_fun)

        self.seg_layout = gui.Horiz()

        self.seg_layout.add_child(self.lock_check)
        self.seg_layout.add_child(self.key_textedit)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(gui.Label("Seg"))
        self.seg_layout.add_child(self.visualize_segments_check)
        self.seg_layout.add_child(self.save_segment_button)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(gui.Label(" C"))
        self.seg_layout.add_child(self.circle_residual_textedit)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(gui.Label(" L"))
        self.seg_layout.add_child(self.line_residual_textedit)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(gui.Label(" T"))
        self.seg_layout.add_child(self.type_textedit)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(gui.Label("Fit"))
        self.seg_layout.add_child(self.visualize_fitting_check)
        self.seg_layout.add_child(self.save_fitting_button)
        self.seg_layout.add_stretch()
        self.seg_layout.add_child(self.reload_parameters_button)

        # self.seg_layout.add_child(self.seg_layout_part1)
        # self.seg_layout.add_child(self.seg_layout_part2)

        self.seg_layout.visible = False

    def lock_check_fun(self, chk):
        if chk:
            self.lock_check_value = True
        else:
            self.lock_check_value = False
        self.my_app.show_lock_count_fun()

    def visualize_segments_check_fun(self, chk):
        if chk:
            self.visualize_segments_check_value = True
        else:
            self.visualize_segments_check_value = False
        seg_key = self.key_textedit.text_value
        self.my_app.visualize_segments(seg_key, chk)

    def visualize_fitting_check_fun(self, chk):
        if chk:
            self.visualize_fitting_check_value = True
        else:
            self.visualize_fitting_check_value = False
        seg_key = self.key_textedit.text_value
        if self.type_textedit.text_value == "circle":
            self.my_app.visualize_circle_fitting(seg_key, chk)
        elif self.type_textedit.text_value == "line":
            self.my_app.visualize_line_fitting(seg_key, chk)
        elif self.type_textedit.text_value == "bspline":
            self.my_app.visualize_bspline_fitting(seg_key, chk)

    def save_segment_fun(self):
        seg_key = self.key_textedit.text_value
        self.my_app.save_segments(seg_key)

    def save_fitting_fun(self):
        seg_key = self.key_textedit.text_value
        seg_type = self.type_textedit.text_value
        if seg_type != "":
            self.my_app.save_fittings(seg_key, seg_type)

    def reload_parameters_fun(self):
        print("reload_parameters_fun")