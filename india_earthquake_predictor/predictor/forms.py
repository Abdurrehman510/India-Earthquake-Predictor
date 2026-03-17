from django import forms

class PredictionForm(forms.Form):
    epicenter_location = forms.CharField(max_length=100, label="Enter Epicenter Location (e.g., Delhi or lat,lon)")
    user_location = forms.CharField(max_length=100, label="Enter Your Location (e.g., Mumbai or lat,lon)", required=False)
    magnitude = forms.FloatField(min_value=1.0, max_value=10.0, label="Enter Earthquake Magnitude")
    use_live_location = forms.BooleanField(required=False, label="Use Live Location (overrides user location input)")