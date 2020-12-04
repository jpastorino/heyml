import os
import sys
import json
import logging
import warnings
import tabulate
import numpy as np
from time import time
from datetime import datetime
from datasets.GraphDatasetMngr import DatasetMngrCSV


# --------------------------------------------------------------------------------------------------------------------
def print_recommendations(ranked):
    """ Prints the recommendations """
    table_width = 240
    elements_width = table_width - 4 - 7 - 15
    header_format = "|{:^7}|{:^14} |{:<" + str(elements_width) + "}|"
    col_format = "|{:^7}|{:>14.8f} |{:<" + str(elements_width) + "}|"

    print("+" + ("-" * (table_width - 2)) + "+")
    print(str("|{:^" + str(table_width - 2) + "}|").format("-- RANKED LIST OF TARGET FEATURES -- "))
    print("|" + ("-" * (table_width - 2)) + "|")
    print(header_format.format("INDEX", "SCORE", " ELEMENTS"))
    print("|" + ("-" * 7) + "+" + ("-" * 14) + "-+" + ("-" * (table_width - 7 - 15 - 4)) + "|")
    rank_i = 0
    for ranked_tuple in ranked:
        print(col_format.format("#" + str(rank_i), ranked_tuple[1], str(ranked_tuple[0])))
        rank_i += 1
    print("-" * table_width)


# --------------------------------------------------------------------------------------------------------------------
def print_recommendations_tabulate(ranked, tab_format="fancy_grid"):
    """Prints the recommendations in tabulated format. https://pypi.org/project/tabulate/"""
    tab_width = 200
    table = []
    rank_i = 0
    for ranked_tuple in ranked:
        table.append(["#" + str(rank_i), ranked_tuple[1], str(ranked_tuple[0])])
        rank_i += 1

    headers = ["Index", "Score", "Elements"]

    print(u'\u2554' + (u'\u2550' * tab_width) + u'\u2557')
    print(u'\u2551' + (("{:^" + str(tab_width) + "}").format("-- RANKED LIST OF TARGET FEATURES --")) + u'\u2551')
    print(u'\u2560' + (u'\u2550' * tab_width) + u'\u2563')

    for line in (tabulate.tabulate(table, headers, tablefmt=tab_format)).split("\n"):
        print(u'\u2551' + (("{:^" + str(tab_width) + "}").format(line)) + u'\u2551')
    print(u'\u255a' + (u'\u2550' * tab_width) + u'\u255d')


# --------------------------------------------------------------------------------------------------------------------
def execute_test_file(datafile_name, log_key, details=False):
    file_name = "../data/" + datafile_name + ".csv"
    with open("../data/" + datafile_name + ".json") as json_file:
        metadata = json.load(json_file)

    logging.basicConfig(filename='../logs/{}_{}.log'.format(datafile_name, log_key), level=logging.DEBUG)

    logging.info("{}=> Start processing file {}.".format(time(), datafile_name))
    start_time = time()
    print("#" * 95)
    print("----- Running File: {:<60} -----\n".format(file_name))

    mngr = DatasetMngrCSV(filename=file_name, metadata=metadata, file_has_header=True)
    mngr.load()
    logging.info("{}=> Dataset loaded.".format(time()))

    n_clusters = np.max([2, (mngr.number_features // 2) + 1])

    print("##### Number of Features: {:2d}.-".format(mngr.number_features))
    print("##### Number of Examples: {:2d}.-".format(mngr.number_data_samples))
    print("##### Number of Clusters: To be optimized.-")

    mngr.generate_clusters(n_clusters=n_clusters, optimize_n_cluster=True)

    if details:
        print("##### Feature Cluster Map: {}.-".format(mngr.cluster_array))
        print("##### Cluster Elements: {}.-".format(mngr.cluster_element_list))

    ranked = mngr.rank_datasets()
    if details:
        print()
        print_recommendations_tabulate(ranked, "fancy_grid")
        print()
        for top_k in [5, 10, 20, 50]:
            print("Evaluation Score is {} for the target in top-{}".format(
                mngr.evaluate(ranked, target_name="target", top_k=top_k), top_k))

    print()
    header = "      Machine Learning problem recommendations for {}".format(file_name)
    print(header)
    print("=" * (len(header) + 6))
    scores = []
    for row in ranked:
        scores.append(row[1])
    scores = np.array(scores)

    scores = np.exp(scores) / sum(np.exp(scores))
    for row_no in range(3):
        if scores[row_no] * 100 >= 1:
            print("With a {:^3.0f}% confidence we recommend that {} can be predicted by a machine learning task.".
                  format(scores[row_no] * 100, str(ranked[row_no][0])))

    print()
    end_time = time()
    print("Pipeline ran in {:.2f} sec.".format(end_time - start_time))
    print("." * 95)
    print("\n\n")
    pass


# ####################################################################################################################
# ####################################################################################################################
# ---------                                         MAIN PROGRAM                                         -------------
# ####################################################################################################################
# ####################################################################################################################
if __name__ == "__main__":
    """ RUNs the Workflow for one file. """
    np.set_printoptions(linewidth=320)

    if not sys.warnoptions:
        warnings.simplefilter("ignore") # Ignore warnings

    files = []
    for file in os.listdir("../data"):
        if file.endswith(".csv"):
            files.append(file[:-4])

    files.sort()
    for i, file in enumerate(files):
        print("{} - {}".format(i + 1, file))
    print("-" * 30)
    selection = int(input("Choose file to process [1-{}]:".format(len(files))))

    if not selection >= 1 and selection <= len(files):
        print("Invalid Selection. Program Terminated.")
        exit(1)

    filename = files[selection - 1]
    print("Processing {}".format(filename))
    print()

    execute_test_file(filename, datetime.today().strftime("%Y.%m.%d.%H.%M.%S"), details=True)
