from django.conf.urls import url
from Image_Style_Transfer_AI.views import HomeView
from django.conf import settings
from django.conf.urls.static import static
from . import views
urlpatterns = [
    url(r'^$', HomeView.as_view(), name="home"),
    url(r'^json$', views.json, name="json"),
    url(r'^logout$', views.logout, name="logout")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
