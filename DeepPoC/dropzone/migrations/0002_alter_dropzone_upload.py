# Generated by Django 3.2.5 on 2021-07-26 13:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dropzone', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dropzone',
            name='upload',
            field=models.ImageField(upload_to='videos/'),
        ),
    ]