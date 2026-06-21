"""
This is the part where dataset construction happens
"""
from datasets import load_dataset

def main():

    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")
    print(dataset)


if __name__ == "__main__":
    main()