from reader import Reader
from extractor import Extractor
from selector import Selector
from classifier import Classifier
from Enums.Types import Types
from evaluator import Evaluator

class Service(object):

    RUN_READER = False
    RUN_EXTRACTOR = False
    RUN_SELECTOR = False
    RUN_CLASSIFIER = True

    BASE_PATH = '/Volumes/My Passport/TCC/WESAD2/'
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
    SELECTOR_ALL_SIGNS = False

    CLASSIFICATION_TIMES = 2

    def run(self):
        if (self.RUN_READER):
            read = Reader()
            read.execute(self.BASE_PATH, self.READER_TYPES, self.BASE_SUBJECTS)

        if (self.RUN_EXTRACTOR):
            extract = Extractor()
            extract.execute(self.BASE_PATH, self.BASE_WINDOW, self.WINDOW_OVERLAP, self.BASE_SUBJECTS)

        if (self.RUN_SELECTOR):
            selection_results = {}
            select = Selector()
            for st in self.SELECTOR_SELECTION_TYPE:
                for sig in self.SELECTOR_SIGNALS:
                    selection_results[sig] = []
                    selection_results[sig] = select.execute(self.BASE_PATH, sig, self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, st, self.SELECTOR_ALL_SIGNS)
                    if (self.SELECTOR_ALL_SIGNS):
                        break
                print('Results = ', st)
                print('RATIO, STD')
                print(selection_results)
            

        if (self.RUN_CLASSIFIER):
            self.individual('ecg', 'pca', 1)
            self.individual('ecg', 'lda', 2)
            self.individual('ecg', '', 3)
            self.individual('eda', 'pca', 1)
            self.individual('eda', 'lda', 2)
            self.individual('eda', '', 3)
            self.individual('emg', 'pca', 1)
            self.individual('emg', 'lda', 2)
            self.individual('emg', '', 3)
            self.individual('resp', 'pca', 1)
            self.individual('resp', 'lda', 2)
            self.individual('resp', '', 3)


            
    def individual(self, signal, selection, number):
        print('Begining ' + str(number) + ' - ' + signal + ' ' + selection + ' ')
        classification = Classifier()
        evaluate = Evaluator()
        i = 0
        predicts_rf = []
        predicts_clf = []
        predicts_nbrs = []
        predicts_shooter = []
        testings = []
        for i in range(self.CLASSIFICATION_TIMES):
            print('times = ', i)
            predicts_rf.insert(i, [])
            predicts_clf.insert(i, [])
            predicts_nbrs.insert(i, [])
            predicts_shooter.insert(i, [])
            testings.insert(i, [])
            predicts_rf[i], predicts_clf[i], predicts_nbrs[i], predicts_shooter[i], testings[i] = classification.execute(self.BASE_PATH, signal, self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, selection, ['svm', 'forest', 'knn', 'shooter'], False, i)

        print('\n\n\n --------------------------------')

        evaluate.report(self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES, testings, predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, '/Volumes/My Passport/TCC/Resultados/' + str(number) + '_' + signal + '.txt')

from service import Service

if __name__ == "__main__":
    print("Process started!")

    serv = Service()
    serv.run()

    