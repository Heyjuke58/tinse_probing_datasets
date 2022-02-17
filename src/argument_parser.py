import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--size", type=int, dest="size", default=10000, help="Size of the generated dataset.",
    )
    parser.add_argument(
        "-sq",
        "--samples_per_query",
        type=int,
        dest="samples_per_query",
        default=5,
        help="Determines the maximumn number of passage samples with the same query in the generated dataset.",
    )
    parser.add_argument(
        "-src",
        "--source",
        type=str,
        dest="source",
        default="msmarco",
        help="Source to add in the info of the dataset",
    )
    parser.add_argument(
        "-ph",
        "--port_http",
        type=str,
        dest="port_http",
        default="12375",
        help="Http port for elasticsearch container",
    )
    parser.add_argument(
        "-pt",
        "--port_tcp",
        type=str,
        dest="port_tcp",
        default="12376",
        help="Tcp port for elasticsearch container",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        dest="tasks",
        default="bm25,semsim,ner,tf",
        help="Tasks to run. Possible tasks are: ['bm25', 'semsim', 'ner', 'corefres', 'factchecking']. Should be comma seperated",
    )
    parser.add_argument(
        "-sp",
        "--sample_path",
        type=str,
        dest="sample_path",
        default=None,
        help="""Reuse an existing sample of a dataset. You need to specify the name of the file in ./datasets/samples/.
            Every time a dataset is newly sampled it is saved in csv format. Naming format: {src}_{size}_{samples per query}_{timestamp}.csv
            If set --size and --samples_per_query are ignored.""",
    )
    parser.add_argument(
        "--split",
        type=str,
        dest="split",
        default="70,15,15",
        help="",
    )

    args = parser.parse_args()

    assert sum(list(map(int, args.split.split(',')))) == 100, "Not a valid train/val/test split. Must add up to 100 like 70,15,15."

    return args