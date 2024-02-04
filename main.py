import requests
import numpy as np
from scipy.interpolate import griddata
import folium
from folium.plugins import HeatMap
import pandas as pd

url = 'http://127.0.0.1:5000/execute_sql'

data = {
    'query': 'SELECT "CLIENT_LATITUDE", "CLIENT_LONGITUDE", "LATENCY" FROM android_extracts_all'
}

response = requests.post(url, json=data)
if response.status_code == 200:
    data = response.json()
    print("Dados recebidos com sucesso!")

    latitud = [feature['geometry']['coordinates'][1] for feature in data['features']]
    longitud = [feature['geometry']['coordinates'][0] for feature in data['features']]
    intensity = [feature['properties']['intensity'] for feature in data['features']]
    
    minIntensity = min(intensity)
    maxIntensity = max(intensity)

    normalized_intensity = [(i - minIntensity) / (maxIntensity - minIntensity) for i in intensity]

    grid_lat, grid_lon = np.mgrid[min(latitud):max(latitud):100j, min(longitud):max(longitud):100j]
    print(grid_lon)
    grid_intensity = griddata((latitud, longitud), normalized_intensity, (grid_lat, grid_lon), method='linear')

    flat_grid_intensity = grid_intensity.flatten()
    normalized_grid_intensity = [(i - np.nanmin(flat_grid_intensity)) / (np.nanmax(flat_grid_intensity) - np.nanmin(flat_grid_intensity)) for i in flat_grid_intensity]

    interpolated_heatmap_data = [[grid_lat.flatten()[i], grid_lon.flatten()[i], normalized_grid_intensity[i]] for i in range(len(normalized_grid_intensity)) if not np.isnan(normalized_grid_intensity[i])]
    print(flat_grid_intensity)
    original_heatmap_data = [[lat, lon, inten] for lat, lon, inten in zip(latitud, longitud, intensity)]
    #print(interpolated_heatmap_data)
    #combined_heatmap_data = original_heatmap_data + interpolated_heatmap_data

    map = folium.Map(location=[-22.9068, -43.1729], zoom_start=10)

    HeatMap(interpolated_heatmap_data).add_to(map)

    # Exportar para Excel, se necessário
    df_cti = pd.DataFrame({'Flat Grid Intensity': normalized_grid_intensity})
    df_cti.to_excel('intensities_cti.xlsx', index=False)

    map.save('heatmap.html')
else:
    print("Erro na requisição:", response.status_code)