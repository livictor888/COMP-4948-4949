from django.urls import path, register_converter
from .views import homePageView, homePost, results

# Custom float converter
class FloatConverter:
    regex = '[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)'

    def to_python(self, value):
        return float(value)

    def to_url(self, value):
        return f'{value:.2f}'

register_converter(FloatConverter, 'float')

urlpatterns = [
    path('', homePageView, name='home'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:length>/<int:margin_low>/<int:margin_up>/<int:diagonal>/<int:height_left>/<int:height_right>', results, name='results'),
]
