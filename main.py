from reader import Reader
from extractor import Extractor
from selector import Selector
from shooter import Shooter
from Enums.Types import Types

RUN_READER = True
RUN_EXTRACTOR = False
RUN_SELECTOR = False
RUN_SHOOTER = False

# Reader Variables
READER_PATH = '/Volumes/My Passport/TCC/WESAD/TESTE/'
READER_TYPES = [
    Types.BASELINE.value,
    Types.STRESS.value,
    Types.AMUSEMENT.value,
]
READER_SUBJECTS = [2]

# Extractor Variables


# Selector Variables


# Shooter Variables



if __name__ == "__main__":
    if (RUN_READER):
        read = Reader()
        read.execute(READER_PATH, READER_TYPES, READER_SUBJECTS)

    if (RUN_EXTRACTOR):
        extract = Extractor
        extract.execute()

    if (RUN_SELECTOR):
        select = Selector
        select.execute()

    if (RUN_SHOOTER):
        shoot = Shooter
        shoot.execute()