import matplotlib.pyplot as plt

# Predictor variables and their Chi-Square scores
predictors = ['length', 'margin_low', 'margin_up', 'diagonal', 'height_right', 'height_left']
scores = [7.308, 86.26, 9.395, 0.01431, 0.3599, 0.1864]

# Plot the bar graph
plt.bar(predictors, scores)

# Set the title and axis labels
plt.title('Chi-Square Scores of Predictor Variables')
plt.xlabel('Predictor Variables')
plt.ylabel('Chi-Square Scores')

# Show the plot
plt.show()