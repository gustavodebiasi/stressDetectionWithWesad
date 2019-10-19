import numpy as np
import math  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class Evaluator(object):

    def report(self, testings, predictions):
        i = 0
        sensibility = []
        specificity = []
        matrix = []
        for i in range(len(predictions)):
            matrix.insert(i, self.matrix_confusion(predictions[i], testings[i]))
            sensibility.insert(i, self.calc_sensibility(matrix[i]))
            specificity.insert(i, self.calc_specificity(matrix[i]))

        print(matrix)
        print(sensibility)
        print(specificity)

    def matrix_confusion(self, predictions, testing):
        return confusion_matrix(testing, predictions)

    def calc_sensibility(self, confusion_matrix):
        return (confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0]))

    def calc_specificity(self, confusion_matrix):
        return (confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1]))

    def print_results(self, predictions, testing_labels):
        # print('Predictions = ', predictions)
        print('acuracy= ', accuracy_score(testing_labels, predictions))
        matrix = confusion_matrix(testing_labels, predictions)
        print('matrix = ', matrix)
        print('sensibility= ', self.calc_sensibility(matrix))
        print('specificity= ', self.calc_specificity(matrix))
        print('--------------------------------')

    def save_results(self, base_path, subject, signal, predicts_rf2, predicts_clf2, predicts_nbrs2, new_results, testing_labels, times):
        all_results = []
        matrix_rf = confusion_matrix(testing_labels, predicts_rf2)
        sensibility_rf = self.calc_sensibility(matrix_rf)
        specificity_rf = self.calc_specificity(matrix_rf)
        matrix_clf = confusion_matrix(testing_labels, predicts_clf2)
        sensibility_clf = self.calc_sensibility(matrix_clf)
        specificity_clf = self.calc_specificity(matrix_clf)
        matrix_nbrs = confusion_matrix(testing_labels, predicts_nbrs2)
        sensibility_nbrs = self.calc_sensibility(matrix_nbrs)
        specificity_nbrs = self.calc_specificity(matrix_nbrs)
        matrix_shoot = confusion_matrix(testing_labels, new_results)
        sensibility_shoot = self.calc_sensibility(matrix_shoot)
        specificity_shoot = self.calc_specificity(matrix_shoot)
        all_results.append([
            sensibility_rf, specificity_rf, sensibility_clf, specificity_clf, sensibility_nbrs, specificity_nbrs, sensibility_shoot, specificity_shoot
        ])
        
        np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/results' + str(times) + '.txt', all_results, fmt="%s")
        matrixs = [
            self.just_numbers(matrix_rf),
            self.just_numbers(matrix_clf),
            self.just_numbers(matrix_nbrs),
            self.just_numbers(matrix_shoot)
        ]
        np.savetxt(base_path + 'S' + str(subject) + '/data/chest_' + signal + '/matrix' + str(times) + '.txt', matrixs, fmt="%s")

    def just_numbers(self, matrix):
        new_matrix = ''
        for m in matrix:
            for n in m:
                new_matrix += ' ' + str(n)
        
        return new_matrix

    def calc_med(self, base_path, signal, subjects, times):
        for sub in subjects:
            # print('sujeito = ', sub)
            all_data = []
            i = 0
            for i in range(times):
                all_data.append(np.asarray(np.loadtxt(base_path + 'S' + str(sub) + '/data/chest_' + signal + '/results' + str(i) + '.txt')))

            meds = []
            k = 0
            for k in range(8):
                j = 0
                med = 0
                for j in range(len(all_data)):
                    if (not math.isnan(all_data[j][k])):
                        med += all_data[j][k]

                meds.extend([(med) / len(all_data)])

            np.savetxt(base_path + 'S' + str(sub) + '/data/chest_' + signal + '/all_results_' + str(times) + '.txt', meds, fmt="%f")

    # def calc_matrix_med(self, base_path, signal, subjects, times):
    #     classifications = {
    #         'svm': [0, 0, 0, 0],
    #         'forest': [0, 0, 0, 0],
    #         'knn': [0, 0, 0, 0],
    #         'shoo': [0, 0, 0, 0],
    #     }

    #     for sub in subjects:
    #         print('subject = ', sub)
    #         i = 0
    #         for i in range(times):
    #             file = np.asarray(np.loadtxt(base_path + 'S' + str(sub) + '/data/chest_' + signal + '/matrix' + str(i) + '.txt'))
    #             classifications['forest'][0] += file[0][0]
    #             classifications['forest'][1] += file[0][1]
    #             classifications['forest'][2] += file[0][2]
    #             classifications['forest'][3] += file[0][3]
    #             classifications['svm'][0] += file[1][0]
    #             classifications['svm'][1] += file[1][1]
    #             classifications['svm'][2] += file[1][2]
    #             classifications['svm'][3] += file[1][3]
    #             classifications['knn'][0] += file[2][0]
    #             classifications['knn'][1] += file[2][1]
    #             classifications['knn'][2] += file[2][2]
    #             classifications['knn'][3] += file[2][3]
    #             classifications['shoo'][0] += file[3][0]
    #             classifications['shoo'][1] += file[3][1]
    #             classifications['shoo'][2] += file[3][2]
    #             classifications['shoo'][3] += file[3][3]
        
    #     classifications = self.calc_media_classifications(classifications, times)

    #     new_array = []
    #     new_array.append(classifications['forest'])
    #     new_array.append(classifications['svm'])
    #     new_array.append(classifications['knn'])
    #     new_array.append(classifications['shoo'])

    #     np.savetxt(base_path + '/resultados_totais_' + str(times) + '.txt', new_array, fmt="%f")

    # def calc_media_classifications(self, classifications, times):
    #     return {
    #         'forest': [
    #             (classifications['forest'][0] / times),
    #             (classifications['forest'][1] / times),
    #             (classifications['forest'][2] / times),
    #             (classifications['forest'][3] / times),
    #         ],
    #         'svm': [
    #             (classifications['svm'][0] / times),
    #             (classifications['svm'][1] / times),
    #             (classifications['svm'][2] / times),
    #             (classifications['svm'][3] / times),
    #         ],
    #         'knn': [
    #             (classifications['knn'][0] / times),
    #             (classifications['knn'][1] / times),
    #             (classifications['knn'][2] / times),
    #             (classifications['knn'][3] / times),
    #         ],
    #         'shoo': [
    #             (classifications['shoo'][0] / times),
    #             (classifications['shoo'][1] / times),
    #             (classifications['shoo'][2] / times),
    #             (classifications['shoo'][3] / times),
    #         ],
    #     }


from evaluator import Evaluator

if __name__ == '__main__':
    evaluate = Evaluator()
