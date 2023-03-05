import numpy as np
                           # Pred1    # Pred2
testArray = np.asarray( [[[0.6, 0.4],[0.2, 0.8]],  # Predictions model 1
                         [[0.1, 0.9],[0.8, 0.2]],  # Predictions model 2
                         [[0.9, 0.1],[0.3, 0.7]],  # Predictions model 3
                         [[0.7, 0.3],[0.6, 0.4]]]) # Predictions model 4
summed = np.sum(testArray, axis=0)
print("\nsummed: ")
print(summed)
