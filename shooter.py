class Shooter(object):
    def choose(self, first_classifier, second_classifier, third_classifier):
        results = []
        i = 0
        for i in range(len(first_classifier)):
            stress = 0
            no_stress = 0
            if (first_classifier[i] == 1.0):
                no_stress += 1
            else:
                stress += 1

            if (second_classifier[i] == 1.0):
                no_stress += 1
            else:
                stress += 1

            if (third_classifier[i] == 1.0):
                no_stress += 1
            else:
                stress += 1

            if (stress > no_stress):
                results.append(2.0)
            else:
                results.append(1.0)
        
        return results


