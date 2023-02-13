from django.db import models
from colorfield.fields import ColorField

class Tag(models.Model):
    name = models.CharField(max_length=100)
    color_hex = ColorField(format="hexa")
    icon_link = models.CharField(max_length=250)
    def __str__(self):
        return f"{self.name}"

class Project(models.Model):
    #Title text
    title_text = models.CharField(max_length=350)
    #link to a .txt, .md, or .html file w/ blog body
    body_text_link = models.CharField(max_length=3000)
    #Tags
    tag_1 = models.ForeignKey(Tag, null=True, blank=True, on_delete=models.SET_NULL, related_name="p_tag_1")
    tag_2 = models.ForeignKey(Tag, null=True, blank=True, on_delete=models.SET_NULL, related_name="p_tag_2")
    tag_3 = models.ForeignKey(Tag, null=True, blank=True, on_delete=models.SET_NULL, related_name="p_tag_3")    

    def __str__(self):
        return f"{self.title_text}]"

