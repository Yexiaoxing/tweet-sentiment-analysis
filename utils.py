import pickle
from datetime import datetime

bcolors = dict(
    HEADER='\033[95m',
    OKBLUE='\033[94m',
    OKGREEN='\033[92m',
    WARNING='\033[93m',
    FAIL='\033[91m',
    ENDC='\033[0m',
    BOLD='\033[1m',
    UNDERLINE='\033[4m'
)


def print_messsage(color: str, heading: str, body: str):
    print("{}[{}]{} ({}) {}".format(bcolors[color], heading,
                                    bcolors["ENDC"], datetime.now().strftime('%H:%M:%S'), body))


info = lambda *body: print_messsage("OKGREEN", "INFO", " ".join(body))
warning = lambda *body: print_messsage("WARNING", "WARN", " ".join(body))
error = lambda *body: print_messsage("FAIL", "ERRO", " ".join(body))


def pickling(filename: str, variable: object):
    """Pickle an variable.

    Arguments:
        filename {str} -- Filename to save
        variable {object} -- Variable to pickle
    """

    with open(filename, "wb") as file:
        pickle.dump(variable, file)
