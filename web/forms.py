from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=50, required=True)
    last_name = forms.CharField(max_length=50, required=False)

    class Meta:
        User._meta.get_field('email')._unique = True
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class AudioUploadForm(forms.Form):
    audio_file = forms.FileField(
        label='Audio File',
        help_text='Supported formats: WAV, MP3, M4A, OGG, FLAC (max 200MB)',
        widget=forms.FileInput(attrs={'accept': 'audio/*', 'class': 'form-control'})
    )
    max_speakers = forms.ChoiceField(
        choices=[(2, '2 Speakers'), (3, '3 Speakers'), (4, '4 Speakers')],
        initial=4,
        label='Maximum Speakers',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def clean_audio_file(self):
        f = self.cleaned_data['audio_file']
        allowed = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.opus']
        import os
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in allowed:
            raise forms.ValidationError(f'Unsupported file format. Allowed: {", ".join(allowed)}')
        if f.size > 200 * 1024 * 1024:
            raise forms.ValidationError('File size must be under 200MB.')
        return f
