from django.db import models

# Create your models here.

class dropzone(models.Model):
    upload = models.ImageField(upload_to='videos/')

    def __str__(self):
        return str(self.pk)