import requests
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import folium
from folium.plugins import HeatMap
import pandas as pd

url = 'http://127.0.0.1:5000/execute_sql'

data = {
    'query': 'SELECT "CLIENT_LATITUDE", "CLIENT_LONGITUDE", "LATENCY" FROM android_extracts_all WHERE "POST_CONNECTION_TYPE"=24'
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

    points = np.column_stack((latitud, longitud))

    values = np.array(normalized_intensity)
    cti = CloughTocher2DInterpolator(points, values)
    grid_lat, grid_lon = np.mgrid[min(latitud):max(latitud):100j, min(longitud):max(longitud):100j]

    # Gerar pontos na grade para interpolação
    grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))

    # Interpolar os dados na grade
    grid_intensity_cti = cti(grid_points)

    # Achatar a matriz de intensidade para uso no HeatMap
    flat_grid_intensity_cti = grid_intensity_cti.flatten()
    print(flat_grid_intensity_cti)
    interpolated_heatmap_data_cti = [[grid_lat.flatten()[i], grid_lon.flatten()[i], flat_grid_intensity_cti[i]] for i in range(len(flat_grid_intensity_cti)) if not np.isnan(flat_grid_intensity_cti[i])]
    
    # Exportar para Excel, se necessário
    df_cti = pd.DataFrame({'Flat Grid Intensity': flat_grid_intensity_cti})
    df_cti.to_excel('intensities_cti.xlsx', index=False)

    # Criar o mapa com Folium
    map_cti = folium.Map(location=[-22.9068, -43.1729], zoom_start=10)
    HeatMap(interpolated_heatmap_data_cti).add_to(map_cti)
    map_cti.save('heatmap_cti.html')

else:
    print("Erro na requisição:", response.status_code)
