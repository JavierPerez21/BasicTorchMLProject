import argparse

parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--testing', default=0.8, help='data path')
args = parser.parse_args()
print(args.testing)