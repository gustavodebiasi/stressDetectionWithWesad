from reader import Reader
from extractor import Extractor
from selector import Selector
from classifier import Classifier
from Enums.Types import Types

class Service(object):

    RUN_READER = False
    RUN_EXTRACTOR = False
    RUN_SELECTOR = False
    RUN_CLASSIFIER = True

    BASE_PATH = '/Volumes/My Passport/TCC/WESAD/'
    BASE_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    BASE_WINDOW = 20
    WINDOW_OVERLAP = True

    # Reader Variables
    READER_TYPES = [
        Types.BASELINE.value,
        Types.STRESS.value
    ]

    # Selector Variables
    SELECTOR_SIGNALS = ['emg', 'resp', 'eda', 'ecg']
    SELECTOR_SELECTION_TYPE = ['pca', 'lda']

    CLASSIFICATION_TIMES = 500

    def run(self):
        if (self.RUN_READER):
            read = Reader()
            read.execute(self.BASE_PATH, self.READER_TYPES, self.READER_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            extract = Extractor()
            extract.execute(self.BASE_PATH, self.BASE_WINDOW, self.WINDOW_OVERLAP, self.BASE_SUBJECTS)

        if (self.RUN_SELECTOR):
            select = Selector()
            for sig in self.SELECTOR_SIGNALS:
                for st in self.SELECTOR_SELECTION_TYPE:
                    select.execute(self.BASE_PATH, sig, self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, st)

        if (self.RUN_CLASSIFIER):
            classification = Classifier()
            for i in range(self.CLASSIFICATION_TIMES):
                classification.execute(self.BASE_PATH, 'ecg', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, 'pca', i)
            classification.calc_med(self.BASE_PATH, 'ecg', self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES)

            for i in range(self.CLASSIFICATION_TIMES):
                classification.execute(self.BASE_PATH, 'eda', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, 'pca', i)
            classification.calc_med(self.BASE_PATH, 'eda', self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES)

            for i in range(self.CLASSIFICATION_TIMES):
                classification.execute(self.BASE_PATH, 'emg', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, 'pca', i)
            classification.calc_med(self.BASE_PATH, 'emg', self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES)

            for i in range(self.CLASSIFICATION_TIMES):
                classification.execute(self.BASE_PATH, 'resp', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, 'pca', i)
            classification.calc_med(self.BASE_PATH, 'resp', self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES)
            

from service import Service

if __name__ == "__main__":
    print("Process started!")

    serv = Service()
    serv.run()

    