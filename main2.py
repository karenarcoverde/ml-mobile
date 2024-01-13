import requests
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.interpolate import RBFInterpolator

# Sua URL e consulta SQL
url = 'http://127.0.0.1:5000/execute_sql'
data = {
    'query': 'SELECT "CLIENT_LATITUDE", "CLIENT_LONGITUDE", "LATENCY" FROM android_extracts_all'
}

response = requests.post(url, json=data)
if response.status_code == 200:
    data = response.json()
    print("Dados recebidos com sucesso!")

    # Extrair latitude, longitude e intensidade
    latitudes = np.array([feature['geometry']['coordinates'][1] for feature in data['features']])
    longitudes = np.array([feature['geometry']['coordinates'][0] for feature in data['features']])
    intensities = np.array([feature['properties']['intensity'] for feature in data['features']])

    # Combina latitude e longitude em uma única matriz para entrada na RBFInterpolator
    points = np.column_stack((latitudes, longitudes))

    # Criar uma função de interpolação Rbf
    rbf_func = RBFInterpolator(points, intensities.reshape(-1, 1), kernel='inverse_multiquadric',epsilon='0.2')

    # Criar uma grade regular para interpolação
    grid_lat, grid_lon = np.mgrid[min(latitudes):max(latitudes):100j, min(longitudes):max(longitudes):100j]
    grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))

    # Interpolar os dados na grade
    grid_intensity = rbf_func(grid_points)

    # Preparar os dados para o mapa de calor
    heatmap_data = np.column_stack((grid_points, grid_intensity.ravel()))

    # Criar o mapa com Folium
    map = folium.Map(location=[np.mean(latitudes), np.mean(longitudes)], zoom_start=10)
    HeatMap(heatmap_data).add_to(map)

    # Salvar o mapa
    map.save('heatmap.html')
else:
    print("Erro na requisição:", response.status_code)
