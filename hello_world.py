import argparse
import time

def hello_world(file):
    with open(file, "a+") as f:
        while True:
            f.write("Hello World\n")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f","--FILENAME", required = False, help = "Filename to write to", type = str)
    args = parser.parse_args()

    hello_world(args.FILENAME)