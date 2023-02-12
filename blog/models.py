from django.db import models
# Create your models here.
class BlogPost(models.Model):
    #Title text
    title_text = models.CharField(max_length=350)
    subtitle_text = models.CharField(max_length=200, null=True, blank=True)
    #Release date
    pub_date = models.DateTimeField('date posted')
    #link to a .txt, .md, or .html file w/ blog body
    body_text_link = models.CharField(max_length=50000)
    #Tags
    project_id = models.ForeignKey("projects.Project", null=True, blank=True, on_delete=models.SET_NULL)
    tag_1 = models.ForeignKey("projects.Tag", null=True, blank=True, on_delete=models.SET_NULL)
    tag_2 = models.ForeignKey("projects.Tag", null=True, blank=True, on_delete=models.SET_NULL)
    tag_3 = models.ForeignKey("projects.Tag", null=True, blank=True, on_delete=models.SET_NULL)     

    def __str__(self):
        return f"{self.title_text} [{self.pub_date.date()}]"