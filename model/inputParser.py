import argparse

parser = argparse.ArgumentParser(description="BRATS CNN")
parser.add_argument('data_dir', metavar="data directory", type=str, help="The directory in which the training data is saved")
parser.add_argument('validation_dir', metavar="validation directory", type=str, help="The directory in which the validation data is saved")
parser.add_argument('save_dir', metavar="model destination directory", type=str, help="The directory in which the model will be saved after training")
parser.add_argument('--model', metavar="model file", type=str, nargs='?', help="The hd5 file of the model to load")

def parse_input():
    args = parser.parse_args()
    print(args)
    return args 
