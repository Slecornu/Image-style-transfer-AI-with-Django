from django.db import models

def get_upload_path(self, filename):
    return '{0}'.format(filename)

# Create your models here.
class User(models.Model):
    email = models.CharField(max_length=250)
    password = models.CharField(max_length=250)

class Art(models.Model):
    artist = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    image_one = models.ImageField(upload_to=get_upload_path)
    image_two = models.ImageField(upload_to=get_upload_path)
    image_output = models.ImageField(upload_to=get_upload_path)


