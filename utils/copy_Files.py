import os
import shutil


def copy_Files(src, destination):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copytree(src=src, dst=destination_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.vscode", "*__pycache__"))


if __name__ == "__main__":
    # src = os.path.abspath(__file__)
    # src = '/home/HaotianF/Exp/Ours/baselines/exp_framwork/utils'
    # dst = "/home/HaotianF/Exp/Ours/baselines/exp_framwork/test"
    # copy_Files(src=src, destination=dst)

    src = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
    src = os.path.dirname(os.path.realpath(__file__))
    print(src)
