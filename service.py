from reader import Reader
from extractor import Extractor
from selector import Selector
from classifier import Classifier
from Enums.Types import Types
from evaluator import Evaluator
from shooter import Shooter

class Service(object):

    RUN_READER = False
    RUN_EXTRACTOR = False
    RUN_SELECTOR = False
    RUN_CLASSIFIER = True

    BASE_PATH = '/Volumes/My Passport/TCC/WESAD3/'
    # BASE_SUBJECTS = [8, 9]
    BASE_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    BASE_WINDOW = 20
    WINDOW_OVERLAP = True

    # Reader Variables
    READER_TYPES = [
        Types.BASELINE.value,
        Types.STRESS.value
    ]

    # Selector Variables
    SELECTOR_SIGNALS = ['resp', 'eda', 'ecg']
    SELECTOR_SELECTION_TYPE = ['pca', 'lda', '']
    SELECTOR_ALL_SIGNS = False

    CLASSIFICATION_TIMES = 100

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
            # self.one_classifier_and_decision('lda', 8)
            # self.one_classifier_and_decision('', 9)
            self.individual('ecg', 'pca', 1)
            # self.individual('ecg', 'lda', 2)
            # self.individual('ecg', '', 3)
            self.individual('eda', 'pca', 1)
            # self.individual('eda', 'lda', 2)
            # self.individual('eda', '', 3)
            # self.individual('emg', 'pca', 1)
            # self.individual('emg', 'lda', 2)
            # self.individual('emg', '', 3)
            self.individual('resp', 'pca', 1)                                                    
            # self.individual('resp', 'lda', 2)
            # self.individual('resp', '', 3)
            self.todos('pca', 4)
            # self.todos('lda', 5)
            # self.todos('', 6)

            self.one_classifier_and_decision('pca', 7)

            # ecg = ['ecg', 'svm']
            # eda = ['eda', 'forest']
            # emg = ['emg', 'svm']
            # resp = ['resp', 'svm']
            

    def one_classifier_and_decision(self, selection, number):
        print('Begining ' + str(number) + ' - todos com decisão ' + selection + ' ')
        shoot2 = Shooter()

        ecg_predicts_rf, ecg_predicts_clf, ecg_predicts_nbrs, ecg_predicts_shooter, ecg_testings = self.individual('ecg', selection, 0, True, ['svm'])
        eda_predicts_rf, eda_predicts_clf, eda_predicts_nbrs, eda_predicts_shooter, eda_testings = self.individual('eda', selection, 0, True, ['forest'])
        # emg_predicts_rf, emg_predicts_clf, emg_predicts_nbrs, emg_predicts_shooter, emg_testings = self.individual('emg', selection, 0, True, ['svm'])
        resp_predicts_rf, resp_predicts_clf, resp_predicts_nbrs, resp_predicts_shooter, resp_testings = self.individual('resp', selection, 0, True, ['svm'])

        # decision_predicts = []
        # i = 0
        # j = 0 
        # for i in range(self.CLASSIFICATION_TIMES):
        #     decision_predicts.insert(i, [])
        #     for j in range(len(self.BASE_SUBJECTS)):
        #         decisao = shoot2.choose(ecg_predicts_clf[i][j], eda_predicts_rf[i][j], emg_predicts_clf[i][j])
        #         decision_predicts[i].insert(j, decisao)

        # evaluate = Evaluator()
        # evaluate.report(self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES, ecg_testings, ecg_predicts_clf, eda_predicts_rf, emg_predicts_clf, decision_predicts, '/Volumes/My Passport/TCC/Resultados3/TODOS_' + str(number) + '_' + selection + '_1.csv')

        decision_predicts2 = []
        i = 0
        j = 0 
        for i in range(self.CLASSIFICATION_TIMES):
            decision_predicts2.insert(i, [])
            for j in range(len(self.BASE_SUBJECTS)):
                decisao = shoot2.choose(ecg_predicts_clf[i][j], eda_predicts_rf[i][j], resp_predicts_clf[i][j])
                decision_predicts2[i].insert(j, decisao)

        evaluate = Evaluator()
        evaluate.report(self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES, ecg_testings, ecg_predicts_clf, eda_predicts_rf, resp_predicts_clf, decision_predicts2, '/Volumes/My Passport/TCC/Resultados3/TODOS_' + str(number) + '_' + selection + '_2.csv')

            
    def individual(self, signal, selection, number, retorno = False, classificador = ['svm', 'forest', 'knn', 'shooter']):
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
            predicts_rf[i], predicts_clf[i], predicts_nbrs[i], predicts_shooter[i], testings[i] = classification.execute(self.BASE_PATH, signal, self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, selection, classificador, False, i)

        if (retorno):
            return predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, testings
        print('\n\n\n --------------------------------')

        evaluate.report(self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES, testings, predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, '/Volumes/My Passport/TCC/Resultados3/' + signal + '_' + str(number) + '.csv')

    def todos(self, selection, number):
        print('Begining ' + str(number) + ' - TODOS ' + selection + ' ')
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
            predicts_rf[i], predicts_clf[i], predicts_nbrs[i], predicts_shooter[i], testings[i] = classification.execute(self.BASE_PATH, 'ecg', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, selection, ['svm', 'forest', 'knn', 'shooter'], True, i)
            # predicts_rf[i], predicts_clf[i], predicts_nbrs[i], predicts_shooter[i], testings[i] = classification.execute(self.BASE_PATH, 'ecg', self.BASE_SUBJECTS, self.BASE_WINDOW, self.WINDOW_OVERLAP, selection, ['svm', 'forest'], True, i)

        print('\n\n\n --------------------------------')

        evaluate.report(self.BASE_SUBJECTS, self.CLASSIFICATION_TIMES, testings, predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, '/Volumes/My Passport/TCC/Resultados3/TODOS_' + str(self.BASE_WINDOW) + '_Over-10_' + str(number) + '.csv')

from service import Service

if __name__ == "__main__":
    print("Process started!")

    serv = Service()
    serv.run()

    