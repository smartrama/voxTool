__author__ = 'iped'

from view.pyloc_view import PyLocControl

if __name__ == '__main__':
    controller = PyLocControl('/Users/iped/PycharmProjects/voxTool/R1170J_CT_combined.nii.gz')
    controller.exec_()