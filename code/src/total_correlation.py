import os
import sys
import json
import logging
import tabulate
import warnings
import numpy as np
from time import time
from datetime import datetime
from datasets.TCDatasetMngr import DatasetMngrCSV


# --------------------------------------------------------------------------------------------------------------------
def print_recommendations(ranked):
    """ Prints the recommendations """
    table_width = 240
    elements_width = table_width - 4 -7 -15
    header_format = "|{:^7}|{:^14} |{:<" + str(elements_width) + "}|"
    col_format    = "|{:^7}|{:>14.8f} |{:<" + str(elements_width) + "}|"

    print("+" +("-" * (table_width-2)) + "+" )
    print(str("|{:^"+str(table_width-2)+"}|").format("-- RANKED LIST OF TARGET FEATURES -- "))
    print("|" + ("-" * (table_width-2))+"|")
    print(header_format.format("INDEX", "SCORE", " ELEMENTS"))
    print("|"+("-" * 7)+"+"+("-" * 14)+"-+"+("-" * (table_width-7-15-4))+"|")
    i = 0
    for ranked_tuple in ranked:
        print(col_format.format("#" + str(i), ranked_tuple[1], str(ranked_tuple[0])))
        i += 1
    print("-" * table_width)


# --------------------------------------------------------------------------------------------------------------------
def print_recommendations_tabulate(ranked, format ="fancy_grid"):
        """ Prints the recommendations in tabulated format. https://pypi.org/project/tabulate/"""
        table=[]
        i = 0
        for ranked_tuple in ranked:
            table.append(["#" + str(i), ranked_tuple[1], str(ranked_tuple[0])])
            i += 1

        headers = ["Index","Score","Elements"]

        print(u'\u2554' +  (u'\u2550' * 250) + u'\u2557'  )
        print(u'\u2551' + ("{:^250}".format("-- RANKED LIST OF TARGET FEATURES --")) + u'\u2551')
        print(u'\u2560' + (u'\u2550' * 250) + u'\u2563')

        # print(tabulate.tabulate(table, headers, tablefmt=format))
        for line in (tabulate.tabulate(table, headers, tablefmt=format)).split("\n"):
            print(u'\u2551' + ("{:^250}".format(line)) + u'\u2551')
        print(u'\u255a' + (u'\u2550' * 250) + u'\u255d')


# --------------------------------------------------------------------------------------------------------------------
def execute_test(filename_arg, log_key):
    logging.basicConfig(filename='../logs/TC_{}_{}.log'.format(filename_arg,log_key), level=logging.DEBUG)

    logging.info("{}=> Start processing file {}.".format(time(), filename_arg))

    start_time = time()
    filename =  "../data/" + filename_arg + ".csv"
    with open( "../data/" + filename_arg + ".json") as json_file:
        metadata = json.load(json_file)

    print("#" * 95)
    print("----- Running Test File: {:<60} -----\n".format(filename))

    mngr = DatasetMngrCSV(filename=filename, metadata=metadata, file_has_header=False)
    mngr.load()
    logging.info("{}=> Dataset loaded.".format(time()))

    print("##### Number of Features: {:2d}.-".format(mngr.number_features))
    print("##### Number of Examples: {:2d}.-".format(mngr.number_data_samples))

    mngr.rank_by_tc()

    end_time = time()
    print("Test Pipeline ran in {:.2f} sec.".format(end_time - start_time))
    print("." * 95)
    print("\n\n")
    pass


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ RUNs the Workflow for one file. """
    np.set_printoptions(linewidth=320)

    if not sys.warnoptions:
        warnings.simplefilter("ignore") # Ignore warnings

    print("="*80)
    print(f"""{"TOTAL CORRELATION EXPERIMENT":^80}""")
    print("="*80)

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
    # """ RUNs the Workflow for one file. """
    # np.set_printoptions(linewidth=320)

    # # Ignore warnings
    # if not sys.warnoptions: warnings.simplefilter("ignore")

    # if len(sys.argv) == 1:
    #     print("Missing filename. \nUsage: {} filename".format(sys.argv[0]))
    #     raise Exception("Missing filename." )

    # execute_test(sys.argv[1].strip(), datetime.today().strftime("%Y.%m.%d.%H.%M.%S"))
   
    execute_test(filename, datetime.today().strftime("%Y.%m.%d.%H.%M.%S"))




