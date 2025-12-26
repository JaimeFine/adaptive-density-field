import folium
import pandas as pd
from branca.colormap import linear

df = pd.read_csv("outputs/communities.csv")

df[['lon', 'lat']] = df[
    'position'
].str.split('_', expand = True).astype(float)

colors = linear.Set1_03.scale(
    df['module'].min(), df['module'].max()
)
center = [df['lat'].mean(), df['lon'].mean()]
m = folium.Map(location = center, zoom_start = 12)

for _, r in df.iterrows():
    folium.CircleMarker(
        location = [r['lat'], r['lon']],
        radius = 4,
        color = colors(r['module']),
        fill = True,
        fill_opacity = 0.9,
        popup = f"{r['position']} -> module {r['module']}"
    ).add_to(m)

m.save("outputs/communities_map.html")