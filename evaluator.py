import numpy as np
import math  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class Evaluator(object):

    path_save_file = ''
    text_file = ''

    def report(self, subjects, times, testings, predicts_rf, predicts_clf, predicts_nbrs, predicts_shooter, path_save_file):
        self.path_save_file = path_save_file

        if (self.path_save_file != ''):
            self.text_file = open(self.path_save_file, 'w')
            self.text_file.write('\n\nRANDOM FOREST\n')
        else:
            print('\n\nRANDOM FOREST')
        self.calc_classifier(times, subjects, testings, predicts_rf)

        if (self.path_save_file != ''):
            self.text_file.write('--------------------------------------------\n')
            self.text_file.write('\n\nSVM\n')
        else:
            print('--------------------------------------------')
            print('\n\nSVM')
        self.calc_classifier(times, subjects, testings, predicts_clf)

        if (self.path_save_file != ''):
            self.text_file.write('--------------------------------------------\n')
            self.text_file.write('\n\nKNN\n')
        else:
            print('--------------------------------------------')
            print('\n\nKNN')
        self.calc_classifier(times, subjects, testings, predicts_nbrs)

        if (self.path_save_file != ''):
            self.text_file.write('--------------------------------------------\n')
            self.text_file.write('\n\nMajority Voting\n')
        else:
            print('--------------------------------------------')
            print('\n\nMajority Voting')
        self.calc_classifier(times, subjects, testings, predicts_shooter)

        if (self.path_save_file != ''):
            self.text_file.close()

    def calc_classifier(self, times, subjects, testings, predictions):
        i = 0
        j = 0
        sensibility = []
        specificity = []
        matrix = []
        for j in range(len(subjects)):
            matrix.insert(j, [])
            sensibility.insert(j, [])
            specificity.insert(j, [])
            for i in range(times):
                matrix[j].insert(i, self.matrix_confusion(predictions[i][j], testings[i][j]))
                sensibility[j].insert(i, self.calc_sensibility(matrix[j][i]))
                specificity[j].insert(i, self.calc_specificity(matrix[j][i]))

        sensibility_resume = []
        sensibility_std = []
        specificity_resume = []
        specificity_std = []
        for j in range(len(subjects)):
            sensibility_resume.insert(j, np.mean(sensibility[j]))
            sensibility_std.insert(j, np.std(sensibility[j]))
            specificity_resume.insert(j, np.mean(specificity[j]))
            specificity_std.insert(j, np.std(specificity[j]))

        if (self.path_save_file != ''):
            self.text_file.write('RESUME PER SUBJECT\n')
            self.text_file.write('sub ;  sensibility   ;   sensibility_std  ;   specificity  ; specificity_std\n')
        else:
            print('RESUME PER SUBJECT')
            print('sub ;  sensibility   ;   sensibility_std  ;   specificity  ; specificity_std')

        for i in range(len(subjects)):
            texto = format(i, '02') + '  ; ' + self.print_percent(sensibility_resume[i]) + '  ; ' + self.print_percent(sensibility_std[i]) + ' ; ' + self.print_percent(specificity_resume[i]) + ' ; ' + self.print_percent(specificity_std[i])
            if (self.path_save_file != ''):
                self.text_file.write(texto + '\n')
            else:
                print(texto)

        if (self.path_save_file != ''):
            self.text_file.write('sub ;  not stressed   ;   stressed;\n')
        else:
            print('sub ;  not stressed   ;   stressed;')
        for i in range(len(subjects)):
            texto = format(i, '02') + '  ; ' + str(matrix[i][0][0][0] + matrix[i][0][0][1]) + '  ; ' + str(matrix[i][0][1][0] + matrix[i][0][1][1])
            if (self.path_save_file != ''):
                self.text_file.write(texto + '\n')
            else:
                print(texto)

        if (self.path_save_file != ''):
            self.text_file.write('-----------\n')
            self.text_file.write('TOTAL RESUME\n')
            self.text_file.write('sensibility ; ' + str(np.mean(sensibility_resume)) + '\n')
            self.text_file.write('specificity ; ' + str(np.mean(specificity_resume)) + '\n')
        else:
            print('-----------')
            print('TOTAL RESUME')
            print('sensibility ; ', np.mean(sensibility_resume))
            print('specificity ; ', np.mean(specificity_resume))

    def print_percent(self, number):
        return str(number).replace('.', ',')

    def matrix_confusion(self, predictions, testing):
        return confusion_matrix(testing, predictions)

    def calc_sensibility(self, confusion_matrix):
        if (confusion_matrix[0][0] == 0):
            return 0
        return (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]))

    def calc_specificity(self, confusion_matrix):
        if (confusion_matrix[1][1] == 0):
            return 0
        return (confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1]))

from evaluator import Evaluator

if __name__ == '__main__':
    evaluate = Evaluator()
