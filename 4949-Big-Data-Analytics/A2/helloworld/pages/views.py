from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
import pickle
import pandas as pd


def homePageView(request):
    # return request object and specify page.
    return render(request, 'home.html')


def homePost(request):
    # Use request object to extract choice.

    length = -999
    margin_low = -999
    margin_up = -999
    diagonal = -999
    height_right = -999
    height_left = -999

    try:
        # Extract value from request object by control name.
        length = request.POST['length']
        margin_low = request.POST['margin_low']
        margin_up = request.POST['margin_up']
        diagonal = request.POST['diagonal']
        height_left = request.POST['height_left']
        height_right = request.POST['height_right']

        # Crude debugging effort.
        length = int(length)
        margin_low = int(margin_low)
        margin_up = int(margin_up)
        diagonal = int(diagonal)
        height_left = int(height_left)
        height_right = int(height_right)
        print("*** Length: " + str(length))
        print("*** Margin Low: " + str(margin_low))
        print("*** Margin Up: " + str(margin_up))
        print("*** Diagonal: " + str(diagonal))
        print("*** Height Left: " + str(height_left))
        print("*** Height Right: " + str(height_right))

    # Enters 'except' block if integer cannot be created.
    except:
        return render(request, 'home.html', {
            'errorMessage': 'The data submitted is invalid. Please try again.'})
    else:
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('results',
                                            kwargs={'length': length, 'margin_low': margin_low, 'margin_up': margin_up,
                                                    'diagonal': diagonal, 'height_left': height_left,
                                                    'height_right': height_right}, ))


def results(request, length, margin_low, margin_up, diagonal, height_left, height_right):
    print("*** Inside results()")
    # load saved model
    with open('model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)

    # Create a single prediction.
    singleSampleDf = pd.DataFrame(columns=[
        'length', 'margin_low', 'margin_up', 'diagonal', 'height_left', 'height_right'])
    currentLength = float(length)
    currentMarginLow = float(margin_low)
    currentMarginUp = float(margin_up)
    currentDiagonal = float(diagonal)
    currentHeightLeft = float(height_left)
    currentHeightRight = float(height_right)
    print("*** Length: " + str(currentLength))
    print("*** Margin Low: " + str(currentMarginLow))
    print("*** Margin Up: " + str(currentMarginUp))
    print("*** Diagonal: " + str(currentDiagonal))
    print("*** Height Left: " + str(currentHeightLeft))
    print("*** Height Right: " + str(currentHeightRight))

    singleSampleDf = pd.DataFrame.from_records([{
        'length': currentLength,
        'margin_low': currentMarginLow,
        'margin_up': currentMarginUp,
        'diagonal': currentDiagonal,
        'height_left': currentHeightLeft,
        'height_right': currentHeightRight
    }])

    singlePrediction = loadedModel.predict(singleSampleDf)

    print("Single prediction: " + str(singlePrediction))
    return render(request, 'results.html',
                  {'length': length, 'margin_low': margin_low, 'margin_up': margin_up, 'diagonal': diagonal,
                   'height_left': height_left, 'height_right': height_right, 'prediction': singlePrediction})
