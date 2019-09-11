from reader import Reader
# from extractor import Extractor
# from selector import Selector
# from shooter import Shooter
from Enums.Types import Types

RUN_READER = True
RUN_EXTRACTOR = False
RUN_SELECTOR = False
RUN_SHOOTER = False

BASE_PATH = '/Volumes/My Passport/TCC/WESAD/'
BASE_SUBJECTS = [2]
# SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Reader Variables
READER_PATH = BASE_PATH
READER_TYPES = [
    Types.BASELINE.value,
    Types.STRESS.value,
    Types.AMUSEMENT.value,
]
READER_SUBJECTS = BASE_SUBJECTS


# Extractor Variables
EXTRACTOR_PATH = BASE_PATH
WINDOW_SIZE = 20
WINDOW_OVERLAP = False
EXTRACTOR_SUBJECTS = BASE_SUBJECTS

# Selector Variables
SELECTOR_PATH = BASE_PATH

# Classifier Variables
EXTRACTOR_PATH = BASE_PATH
CLASSIFIER_SUBJECTS = BASE_SUBJECTS

# Shooter Variables
EXTRACTOR_PATH = BASE_PATH
CLASSIFIER_SUBJECTS = BASE_SUBJECTS


if __name__ == "__main__":
    print("Inicia processo!")

    if (RUN_READER):
        read = Reader()
        read.execute(READER_PATH, READER_TYPES, READER_SUBJECTS)

    # if (RUN_EXTRACTOR):
    #     extract = Extractor
    #     extract.execute()

    # if (RUN_SELECTOR):
    #     select = Selector
    #     select.execute()

    # if (RUN_SHOOTER):
    #     shoot = Shooter
    #     shoot.execute()