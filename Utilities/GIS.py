import numpy as np

import folium
from folium import plugins

import webbrowser

import shapely
from shapely.geometry import shape
from shapely.algorithms.polylabel import polylabel

basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    )
}

class My_Map():
    def __init__(self, location=[40, -95], zoom_start=5):
        self.m=folium.Map(location=location, zoom_start=zoom_start)

        formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"

        plugins.MousePosition(
            position="topright",
            separator=" | ",
            empty_string="NaN",
            lng_first=True,
            num_digits=20,
            prefix="Coordinates:",
            lat_formatter=formatter,
            lng_formatter=formatter,
        ).add_to(self.m)

        # Add custom basemaps
        basemaps['Google Maps'].add_to(self.m)
        basemaps['Google Satellite'].add_to(self.m)
        basemaps['Google Satellite Hybrid'].add_to(self.m)
        basemaps['Google Terrain'].add_to(self.m)
        basemaps['Esri Satellite'].add_to(self.m)

        # Add fullscreen button
        plugins.Fullscreen().add_to(self.m)

    def show(self, save_file = "folium.html"):
        self.m.save(save_file)
        webbrowser.open(save_file, new=2)  # open in new tab

    def add_LayerControl(self, save_file = "folium.html"):
        # This needs to be added after adding all the layers (otherwise it doesn't work)
        self.m.add_child(folium.LayerControl())
    
def pole_of_inaccessibility(input, accuracy=0.1):
    poles=[]
    areas = []

    # print(type(input))
    if (type(input) == dict): input = shape(input)
        

    if (type(input) == shapely.geometry.multipolygon.MultiPolygon):
        for p in input.geoms:
            areas.append(p.area)
            poles.append(polylabel(p,accuracy))

    elif (type(input) == shapely.geometry.polygon.Polygon):
            areas.append(input.area)
            poles.append(polylabel(input,accuracy))

    return poles[np.argmax(areas)]    