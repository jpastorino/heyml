import numpy as np


# --------------------------------------------------------------------------------------------------------------------
def text_on_box(text: str, size: int):
    """
    Prints the text withing a double-border box of the given size.
    :param text: text to print centered in the box
    :param size:  box size
    :return: None.
    """
    print(u'\u2554' + u'\u2550' * size + u'\u2557')
    print(u'\u2551' + f"""{text:^{size}}""" + u'\u2551')
    print(u'\u255A' + u'\u2550' * size + u'\u255D')
    pass


# --------------------------------------------------------------------------------------------------------------------
def main_menu(files: list) -> str:
    """
    Given a file list, displays the main menu of HeyMl to allow the user selecting the dataset to process.
    :param files: filename lists.
    :return: the selected filename.
    """

    files.sort()

    menu_width = 60
    print(u'\u2554' + u'\u2550' * menu_width + u'\u2557')
    print(u'\u2551' + f"""{"Main Menu":^{menu_width}}""" + u'\u2551')
    print(u'\u2560' + u'\u2550' * menu_width + u'\u2563')
    print(u'\u2551' + f"""{"Available Datasets":^{menu_width}}""" + u'\u2551')
    print(u'\u255F' + u'\u2500' * menu_width + u'\u2562')

    for i, file in enumerate(files):
        print(u'\u2551' + f"""{i+1:3} - {file:{menu_width-6}}""" + u'\u2551')
    print(u'\u255A' + u'\u2550' * menu_width + u'\u255D')

    selection = int(input(u"\u2593"*20+" Choose file to process [1-{}]".format(len(files)) + u"\u25B7" + " "))

    if not selection >= 1 and selection <= len(files):
        print("Invalid Selection. Program Terminated.")
        exit(1)

    filename = files[selection - 1]
    return filename

    pass


# --------------------------------------------------------------------------------------------------------------------
def display_recommendations(ranked: list, file_name: str):
    """
    Prints to Console the Recommendations
    :param ranked: list of computed ranked recommendations
    :param file_name: full relative filename and path of the dataset.
    :return: None
    """

    screen_width = 120
    parsed_filename = file_name.split("/")[-1]
    print()
    print(u'\u250C' + u'\u2500' * screen_width + u'\u2510')
    print(u'\u2502' + f"""{f'''Machine Learning Problem Recommendations for {parsed_filename}''':^{screen_width}}""" + u'\u2502')
    print(u'\u251C' + u'\u2500' * screen_width + u'\u2524')

    scores = []
    for row in ranked:
        scores.append(row[1])
    scores = np.array(scores)

    scores = np.exp(scores) / sum(np.exp(scores))
    for row_no in range(3):
        if scores[row_no] * 100 >= 1:
            rec_text = "With a {:^3.0f}% confidence we recommend that {} can be predicted by a machine learning task."\
                    .format(scores[row_no] * 100, str(ranked[row_no][0]))
            print(u'\u2502' + f"""{rec_text:{screen_width}}"""+ u'\u2502')

    print(u'\u2514' + u'\u2500' * screen_width + u'\u2518')
    pass
