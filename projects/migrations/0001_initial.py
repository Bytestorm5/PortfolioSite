# Generated by Django 4.1.6 on 2023-02-13 03:05

import colorfield.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('color_hex', colorfield.fields.ColorField(default='#FFFFFFFF', image_field=None, max_length=18, samples=None)),
                ('icon_link', models.CharField(max_length=250)),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title_text', models.CharField(max_length=350)),
                ('body_text_link', models.CharField(max_length=3000)),
                ('tag_1', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='p_tag_1', to='projects.tag')),
                ('tag_2', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='p_tag_2', to='projects.tag')),
                ('tag_3', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='p_tag_3', to='projects.tag')),
            ],
        ),
    ]
