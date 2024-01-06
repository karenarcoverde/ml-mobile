import requests
import numpy as np
from scipy.interpolate import griddata
import folium
from folium.plugins import HeatMap

url = 'http://127.0.0.1:5000/execute_sql'

data = {
    'query': 'SELECT "CLIENT_LATITUDE", "CLIENT_LONGITUDE", "LATENCY" FROM android_extracts_all'
}

response = requests.post(url, json=data)

if response.status_code == 200:
    data = response.json()
    print("Dados recebidos com sucesso!")
else:
    print("Erro na requisição:", response.status_code)

latitud = [feature['geometry']['coordinates'][1] for feature in data['features']]
longitud = [feature['geometry']['coordinates'][0] for feature in data['features']]
intensity = [feature['properties']['intensity'] for feature in data['features']]

grid_lat, grid_lon = np.mgrid[min(latitud):max(latitud):100j, min(longitud):max(longitud):100j]

grid_intensity = griddata((latitud, longitud), intensity, (grid_lat, grid_lon), method='cubic')

heatmap_data = [[grid_lat[i][j], grid_lon[i][j], grid_intensity[i][j]] for i in range(100) for j in range(100) if not np.isnan(grid_intensity[i][j])]

map = folium.Map(location=[-22.9068, -43.1729], zoom_start=10)

HeatMap(heatmap_data).add_to(map)

map.save('heatmap.html')
