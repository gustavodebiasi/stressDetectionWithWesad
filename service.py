from reader import Reader
from extractor import Extractor
# from selector import Selector
# from shooter import Shooter
from Enums.Types import Types

class Service(object):

    RUN_READER = False
    RUN_EXTRACTOR = True
    RUN_SELECTOR = False
    RUN_SHOOTER = False

    BASE_PATH = '/Volumes/My Passport/TCC/WESAD/'
    # BASE_SUBJECTS = [2]
    BASE_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    # Reader Variables
    READER_PATH = BASE_PATH
    READER_TYPES = [
        Types.BASELINE.value,
        Types.STRESS.value
    ]
    READER_SUBJECTS = BASE_SUBJECTS

    # Extractor Variables
    EXTRACTOR_PATH = BASE_PATH
    EXTRACTOR_WINDOW_SIZE = 20
    EXTRACTOR_WINDOW_OVERLAP = False
    EXTRACTOR_SUBJECTS = BASE_SUBJECTS

    # Selector Variables
    SELECTOR_PATH = BASE_PATH

    # Classifier Variables
    EXTRACTOR_PATH = BASE_PATH
    CLASSIFIER_SUBJECTS = BASE_SUBJECTS

    # Shooter Variables
    EXTRACTOR_PATH = BASE_PATH
    CLASSIFIER_SUBJECTS = BASE_SUBJECTS

    def run(self):
        if (self.RUN_READER):
            read = Reader()
            read.execute(self.READER_PATH, self.READER_TYPES, self.READER_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            print('3')
            extract = Extractor()
            extract.execute(base_path=self.EXTRACTOR_PATH, window=20, window_overlap=True, subjects=self.EXTRACTOR_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            print('4')
            extract = Extractor()
            extract.execute(base_path=self.EXTRACTOR_PATH, window=20, window_overlap=False, subjects=self.EXTRACTOR_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            print('1')
            extract = Extractor()
            extract.execute(base_path=self.EXTRACTOR_PATH, window=30, window_overlap=True, subjects=self.EXTRACTOR_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            print('2')
            extract = Extractor()
            extract.execute(base_path=self.EXTRACTOR_PATH, window=30, window_overlap=False, subjects=self.EXTRACTOR_SUBJECTS)

        # if (RUN_SELECTOR):
        #     select = Selector
        #     select.execute()

        # if (RUN_SHOOTER):
        #     shoot = Shooter
        #     shoot.execute()

from service import Service

if __name__ == "__main__":
    print("Process started!")

    serv = Service()
    serv.run()

    