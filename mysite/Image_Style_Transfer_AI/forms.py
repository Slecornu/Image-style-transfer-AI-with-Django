from django import forms
from Image_Style_Transfer_AI.models import User

class ImageUploadForm(forms.Form):
    image_one = forms.ImageField(required=True)
    image_two = forms.ImageField(required=True)
    num_of_iterations = forms.IntegerField(min_value=0, initial=50, required=True)

class LoginForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["email", "password"]
        widgets = {
            "email": forms.EmailInput(),
            'password': forms.PasswordInput(),
        }

class SignUpForm(forms.Form):
    email = forms.CharField(widget=forms.EmailInput(), required=True)
    password = forms.CharField(widget=forms.PasswordInput(), required=True)
    password_retyped = forms.CharField(widget=forms.PasswordInput(), required=True)