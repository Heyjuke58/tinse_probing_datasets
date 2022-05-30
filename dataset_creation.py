import logging
import os

from nltk import download

from src.dataset_creator import DatasetCreator
from src.argument_parser import parse_arguments

# set visible devices to -1 since no gpu is needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

download("stopwords")
download("punkt")


def main(args):
    logging.basicConfig(filename="msmarco.log", filemode="w+", level=logging.INFO)

    dc = DatasetCreator(
        args.tasks.split(','),
        args.source,
        args.size,
        args.samples_per_query,
        args.port_http,
        args.port_tcp,
        args.sample_path,
        args.neg_sample_ratio,
        args.split,
        args.id_pairs
    )

    dc.run()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
