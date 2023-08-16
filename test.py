import matplotlib.pyplot as plt

import anesthesia_models
import patient
import predictors

if __name__ == "__main__":
    predictors.test()
    anesthesia_models.test()
    patient.test()
    plt.show()