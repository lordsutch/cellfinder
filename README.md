# cellfinder
code for geolocation of cell sites using data from my SignalDetector
Android application

Dependencies:

Python >= 3.6

> pip3 install scipy pandas folium pyproj haversine

Also tested with PyPy3 >= 5.10, but NumPy calls are really slow in
PyPy3 at present so you're actually better off in standard Python:

https://bitbucket.org/pypy/pypy/issues/2357/numpy-code-50x-slower-than-python
