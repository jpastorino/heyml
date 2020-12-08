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
from heyml_stuff import main_menu, text_on_box, display_recommendations


# --------------------------------------------------------------------------------------------------------------------
def print_ranked_results(ranked, tab_format="fancy_grid"):
    """
    Prints the ranked results in tabulated format. https://pypi.org/project/tabulate/
    :param ranked: Ranked recommendations.
    :param tab_format: tabulate format. default "fancy_grid"
    :return:
    """

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
def process_dataset_file(datafile_name, log_key, details=False):
    """
    Given a dataset, process the file with the HeyML black Box.
    :param datafile_name:  filename in the ../data directory
    :param log_key: postfix to add to the log file for identification (e.g. datetime.)
    :param details: If shows Details True, will print internal ranking.
    :return: None
    """
    file_name = "../data/" + datafile_name + ".csv"
    with open("../data/" + datafile_name + ".json") as json_file:
        metadata = json.load(json_file)

    logging.basicConfig(filename='../logs/{}_{}.log'.format(datafile_name, log_key), level=logging.DEBUG)

    logging.info("{}=> Start processing file {}.".format(time(), datafile_name))
    start_time = time()
    text_on_box("Processing File: "+file_name, 120)
    mngr = DatasetMngrCSV(filename=file_name, metadata=metadata, file_has_header=True)
    mngr.load()
    logging.info("{}=> Dataset loaded.".format(time()))

    n_clusters = np.max([2, (mngr.number_features // 2) + 1])

    # Printing data summary
    print(u"\u2593" * 6 + f'''{f""" Number of Features: {mngr.number_features:2d}""":40}'''+u"\u2593" * 6 )
    print(u"\u2593" * 6 + f'''{f""" Number of Examples: {mngr.number_data_samples:2d}""":40}''' + u"\u2593" * 6)
    # print(u"\u2593" * 6 + f'''{f""" Number of Clusters: To be optimized.""":40}''' + u"\u2593" * 6)

    best_n_cluster = mngr.generate_clusters(n_clusters=n_clusters, optimize_n_cluster=True)
    print(u"\u2593" * 6 + f'''{f""" Optimized Number of Clusters: {best_n_cluster:2d}""":40}''' + u"\u2593" * 6)
    if details:
        print(u"\u2593" * 6 + f'''{f""" Feature Cluster Map: {mngr.cluster_array}""":40}''' + u"\u2593" * 6)
        print(u"\u2593" * 6 + f'''{f""" Cluster Elements: {mngr.cluster_element_list}""":40}''' + u"\u2593" * 6)

    ranked = mngr.rank_datasets()
    if details:
        print()
        print_ranked_results(ranked, "fancy_grid")
        print()
        for top_k in [5, 10, 20, 50]:
            print("Evaluation Score is {} for the target in top-{}".format(
                mngr.evaluate(ranked, target_name="target", top_k=top_k), top_k))

    # ########## PRINTING RECOMMENDATIONS ###########
    display_recommendations(ranked, file_name)

    print()
    end_time = time()
    print("Pipeline ran in {:.2f} sec.".format(end_time - start_time))
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
        warnings.simplefilter("ignore")  # Ignore warnings

    files = []
    for file in os.listdir("../data"):
        if file.endswith(".csv"):
            files.append(file[:-4])

    filename = main_menu(files)

    print(f"{filename} Dataset File Selected")
    print()

    process_dataset_file(filename, datetime.today().strftime("%Y.%m.%d.%H.%M.%S"), details=False)
