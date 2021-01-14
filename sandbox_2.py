

from dvas.environ import path_var


class A:

    def get(self):
        print(path_var.config_dir_path)
