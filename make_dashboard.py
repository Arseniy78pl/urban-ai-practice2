"""Собирает один файл dashboard.html из готовых результатов ноутбука practice02_chernogolovka.ipynb.

Ничего не скачивает и не пересчитывает: читает только файлы из папок data/ и results/.
Запуск (Windows, Command Prompt, в папке проекта):
    .venv\\Scripts\\activate.bat
    python make_dashboard.py
"""

import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUT_FILE = ROOT / "dashboard.html"

# ---------------------------------------------------------------- чтение результатов

blocks = gpd.read_file(RESULTS_DIR / "blocks_all.geojson")
boundary = gpd.read_file(DATA_DIR / "boundary.geojson")
schools = gpd.read_file(RESULTS_DIR / "schools_imputed.geojson")
links = gpd.read_file(RESULTS_DIR / "provision_links.geojson")
clusters = pd.read_csv(RESULTS_DIR / "spacematrix_clusters.csv", index_col=0)
acc_mx = pd.read_csv(RESULTS_DIR / "accessibility_matrix_walk.csv", index_col=0)
summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
sources = json.loads((DATA_DIR / "sources.json").read_text(encoding="utf-8"))

blocks["block_id"] = blocks["block_id"].astype(int)
blocks = blocks.set_index("block_id", drop=False).sort_index()
blocks["site_area_ha"] = blocks["site_area"] / 1e4
utm = blocks.estimate_utm_crs()

LAND_USE_RU = {
    "residential": "жилая", "business": "общественно-деловая", "recreation": "рекреационная",
    "industrial": "производственная", "transport": "транспортная", "special": "специальная",
    "agriculture": "сельскохозяйственная",
}
LAND_USE_COLS = list(LAND_USE_RU)
blocks["land_use_ru"] = blocks["land_use"].map(LAND_USE_RU).fillna("не определено")

# ---------------------------------------------------------------- описание показателей
# kind: num — число (последовательная шкала), share — доля 0–1, cat — категория

METRICS = [
    # key, подпись, единицы, группа, вид, источник/оценка
    ("land_use_ru", "Преобладающая функция", "", "Землепользование", "cat", "данные OSM"),
    *[(c, f"Доля: {LAND_USE_RU[c]}", "доля", "Землепользование", "share", "данные OSM") for c in LAND_USE_COLS],
    ("mean_time_min", "Среднее время до других кварталов", "мин", "Транспорт", "num", "расчёт по графу"),
    ("max_time_min", "Максимальное время до других кварталов", "мин", "Транспорт", "num", "расчёт по графу"),
    ("bld_population", "Население", "чел.", "Здания", "num", "модельная оценка"),
    ("bld_footprint_area", "Площадь пятен застройки", "м²", "Здания", "num", "данные OSM"),
    ("bld_build_floor_area", "Площадь этажей", "м²", "Здания", "num", "частично оценка (этажность)"),
    ("bld_living_area", "Жилая площадь", "м²", "Здания", "num", "модельная оценка"),
    ("bld_non_living_area", "Нежилая площадь", "м²", "Здания", "num", "модельная оценка"),
    ("bld_count", "Число зданий", "шт.", "Здания", "num", "данные OSM"),
    ("sch_count", "Число школ", "шт.", "Школы", "num", "данные OSM"),
    ("sch_capacity", "Вместимость школ", "мест", "Школы", "num", "оценка BlocksNet"),
    ("sch_capacity_imputed", "в т. ч. оценённая вместимость", "мест", "Школы", "num", "оценка BlocksNet"),
    ("prov_demand", "Спрос на места в школах", "мест", "Обеспеченность", "num", "модель"),
    ("prov_demand_within", "Спрос, закрытый в 15 мин", "мест", "Обеспеченность", "num", "модель"),
    ("prov_demand_without", "Спрос, закрытый дальше 15 мин", "мест", "Обеспеченность", "num", "модель"),
    ("prov_demand_left", "Нехватка мест", "мест", "Обеспеченность", "num", "модель"),
    ("prov_capacity_left", "Остаток свободных мест", "мест", "Обеспеченность", "num", "модель"),
    ("prov_capacity_within", "Места, занятые в пределах 15 мин", "мест", "Обеспеченность", "num", "модель"),
    ("prov_capacity_without", "Места, занятые дальше 15 мин", "мест", "Обеспеченность", "num", "модель"),
    ("prov_provision_strong", "Обеспеченность строгая (≤ 15 мин)", "доля", "Обеспеченность", "share", "модель"),
    ("prov_provision_weak", "Обеспеченность мягкая", "доля", "Обеспеченность", "share", "модель"),
    ("fsi", "FSI — плотность застройки", "м²/м²", "Плотность", "num", "расчёт"),
    ("gsi", "GSI — застроенность", "доля", "Плотность", "share", "расчёт"),
    ("l", "L — средняя этажность", "этажей", "Плотность", "num", "расчёт"),
    ("osr", "OSR — открытое пространство", "м²/м²", "Плотность", "num", "расчёт"),
    ("mxi", "MXI — доля жилья", "доля", "Плотность", "share", "расчёт"),
    ("sm_morphotype", "Тип застройки Spacematrix", "", "Типы застройки", "cat", "кластеризация"),
    ("st_morphotype", "Тип застройки «Стрелка»", "", "Типы застройки", "cat", "правила"),
    ("area_accessibility", "Доступность с учётом площади", "мин", "Доступность", "num", "расчёт по графу"),
    ("site_area_ha", "Площадь квартала", "га", "Кварталы", "num", "расчёт"),
]
METRICS = [m for m in METRICS if m[0] in blocks.columns]

# ---------------------------------------------------------------- данные для карты


def clean(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 3)
    return v


map_cols = ["block_id"] + [m[0] for m in METRICS]
blocks_4326 = blocks[map_cols + ["geometry"]].to_crs(4326).copy()
blocks_4326["geometry"] = blocks_4326.geometry.simplify(0.00001)
blocks_geojson = json.loads(blocks_4326.to_json(drop_id=True))
for f in blocks_geojson["features"]:
    f["properties"] = {k: clean(v) for k, v in f["properties"].items()}

schools_4326 = schools.to_crs(utm)
schools_4326["geometry"] = schools_4326.representative_point()
schools_4326 = schools_4326.to_crs(4326)
schools_points = [
    {"lat": p.y, "lon": p.x, "name": n if isinstance(n, str) else "школа без названия", "capacity": clean(c), "src": s}
    for p, n, c, s in zip(schools_4326.geometry, schools_4326["name"], schools_4326["capacity"], schools_4326["capacity_source"])
]
links_geojson = json.loads(links.to_crs(4326).to_json(drop_id=True))
boundary_geojson = json.loads(boundary.to_crs(4326)[["geometry"]].to_json(drop_id=True))

metrics_js = [
    {"key": k, "label": lab, "unit": u, "group": g, "kind": kind, "origin": o}
    for k, lab, u, g, kind, o in METRICS
]

# ---------------------------------------------------------------- графики (Plotly, данные готовятся здесь)

charts = {}

lu_area = blocks.groupby("land_use_ru")["site_area_ha"].sum().sort_values()
charts["landuse"] = {
    "data": [{"type": "bar", "orientation": "h", "x": lu_area.round(1).tolist(), "y": lu_area.index.tolist(),
              "hovertemplate": "%{y}: %{x:.1f} га<extra></extra>", "marker": {"color": "series1"}}],
    "layout": {"xaxis": {"title": "площадь, га"}, "margin": {"l": 170}},
}

top_pop = blocks[blocks["bld_population"] > 0].sort_values("bld_population", ascending=False).head(15)
charts["population"] = {
    "data": [{"type": "bar", "x": [f"кв. {i}" for i in top_pop.index], "y": top_pop["bld_population"].round(0).tolist(),
              "hovertemplate": "%{x}: %{y:.0f} чел.<extra></extra>", "marker": {"color": "series1"}}],
    "layout": {"yaxis": {"title": "чел. (оценка)"}},
}

dem = blocks[blocks["prov_demand"] > 0].sort_values("prov_demand", ascending=False).head(15)
xlab = [f"кв. {i}" for i in dem.index]
charts["provision"] = {
    "data": [
        {"type": "bar", "name": "закрыто в 15 мин", "x": xlab, "y": dem["prov_demand_within"].round(0).tolist(),
         "marker": {"color": "series1"}, "hovertemplate": "%{x}: %{y:.0f} мест<extra>закрыто в 15 мин</extra>"},
        {"type": "bar", "name": "закрыто дальше 15 мин", "x": xlab, "y": dem["prov_demand_without"].round(0).tolist(),
         "marker": {"color": "series2"}, "hovertemplate": "%{x}: %{y:.0f} мест<extra>дальше 15 мин</extra>"},
        {"type": "bar", "name": "нехватка мест", "x": xlab, "y": dem["prov_demand_left"].round(0).tolist(),
         "marker": {"color": "series3"}, "hovertemplate": "%{x}: %{y:.0f} мест<extra>нехватка</extra>"},
    ],
    "layout": {"barmode": "stack", "yaxis": {"title": "мест в школах"}, "bargap": 0.3},
}

built = blocks[blocks["fsi"] > 0]
sm_types = built["sm_morphotype"].fillna("—")
charts["spacematrix"] = {
    "data": [
        {"type": "scatter", "mode": "markers", "name": t, "x": built.loc[sm_types == t, "gsi"].round(3).tolist(),
         "y": built.loc[sm_types == t, "fsi"].round(3).tolist(),
         "text": [f"кв. {i}" for i in built.index[sm_types == t]],
         "marker": {"size": 10, "color": f"cat{k}", "line": {"width": 2, "color": "surface"}},
         "hovertemplate": "%{text}<br>GSI %{x:.2f}, FSI %{y:.2f}<extra>" + t + "</extra>"}
        for k, t in enumerate(sm_types.value_counts().index)
    ],
    "layout": {"xaxis": {"title": "GSI — доля застроенной земли"}, "yaxis": {"title": "FSI — площадь этажей / площадь квартала"},
               "legend": {"orientation": "h", "y": -0.25}},
}

st_counts = blocks["st_morphotype"].dropna().value_counts().sort_values()
charts["strelka"] = {
    "data": [{"type": "bar", "orientation": "h", "x": st_counts.tolist(), "y": st_counts.index.tolist(),
              "hovertemplate": "%{y}: %{x} кварталов<extra></extra>", "marker": {"color": "series1"}}],
    "layout": {"xaxis": {"title": "кварталов"}, "margin": {"l": 210}},
}

aa = blocks[["area_accessibility", "site_area_ha"]].dropna()
charts["area_acc"] = {
    "data": [{"type": "histogram", "x": aa["area_accessibility"].round(1).tolist(), "nbinsx": 15,
              "marker": {"color": "series1", "line": {"width": 2, "color": "surface"}},
              "hovertemplate": "%{x} мин: %{y} кварталов<extra></extra>"}],
    "layout": {"xaxis": {"title": "минут пешком (взвешено по площади)"}, "yaxis": {"title": "кварталов"}},
}

charts["matrix"] = {
    "data": [{"type": "heatmap", "z": acc_mx.round(1).values.tolist(), "x": [f"кв. {c}" for c in acc_mx.columns],
              "y": [f"кв. {c}" for c in acc_mx.index], "colorscale": "seq", "colorbar": {"title": "мин"},
              "hovertemplate": "%{y} → %{x}: %{z:.1f} мин<extra></extra>"}],
    "layout": {"xaxis": {"showticklabels": False}, "yaxis": {"showticklabels": False, "autorange": "reversed"}},
}

# Plotly 2.x ждёт подписи осей в виде {"text": ...}
for ch in charts.values():
    for ax in ("xaxis", "yaxis"):
        t = ch["layout"].get(ax, {}).get("title")
        if isinstance(t, str):
            ch["layout"][ax]["title"] = {"text": t}

# ---------------------------------------------------------------- таблицы


def table(df: pd.DataFrame, cls: str = "", index: bool = False) -> str:
    return df.to_html(classes=f"tbl {cls}", border=0, index=index, na_rep="—", escape=True, float_format=lambda v: f"{v:,.2f}".replace(",", " "))


summary_rows = "".join(
    f'<div class="stat"><div class="stat-v">{html.escape(f"{v:,}".replace(",", " ") if isinstance(v, int) else str(v))}</div>'
    f'<div class="stat-l">{html.escape(k)}</div></div>'
    for k, v in summary.items()
)

all_cols = {m[0]: f"{m[1]}{', ' + m[2] if m[2] else ''}" for m in METRICS}
blocks_table = blocks[["block_id"] + list(all_cols)].rename(columns={"block_id": "квартал", **all_cols})
blocks_table_html = table(blocks_table.round(2), "wide")

clusters_table = clusters.rename(columns={"l": "L", "fsi": "FSI", "mxi": "MXI", "l_interpretation": "этажность",
                                          "fsi_interpretation": "плотность", "mxi_interpretation": "функция",
                                          "morphotype": "морфотип"}).round(2)
clusters_table.insert(0, "кварталов", blocks["sm_cluster"].value_counts().reindex(clusters.index).fillna(0).astype(int))
clusters_table.index.name = "кластер"
clusters_html = table(clusters_table, index=True)

schools_table = pd.DataFrame({
    "школа": [s["name"] for s in schools_points],
    "вместимость, мест": [s["capacity"] for s in schools_points],
    "откуда вместимость": [s["src"] for s in schools_points],
})
schools_html = table(schools_table)

links_table = pd.DataFrame(links.drop(columns="geometry")).rename(
    columns={"from": "квартал спроса", "to": "квартал со школой", "seats": "мест", "time_min": "время, мин"}).round(1)
links_html = table(links_table) if len(links_table) else "<p>Связей нет.</p>"

glossary = pd.DataFrame(METRICS, columns=["поле", "показатель", "единицы", "группа", "вид", "происхождение"]).drop(columns="вид")
glossary_html = table(glossary)

sources_html = "".join(
    f"<li><b>{html.escape(k)}</b>: {html.escape(v.get('source', ''))}, загружено {html.escape(v.get('downloaded', ''))}</li>"
    for k, v in sources.items()
)

# ---------------------------------------------------------------- выводы (по данным)

lu_share = (blocks.groupby("land_use_ru")["site_area"].sum() / blocks["site_area"].sum() * 100).round(0)
top_lu = lu_share.sort_values(ascending=False)
pop_total = blocks["bld_population"].sum()
top3_pop = blocks["bld_population"].nlargest(3).sum() / pop_total if pop_total else 0
conclusions = [
    f"Территория города ({summary['площадь города, км²']} км²) разделена на {len(blocks)} кварталов. Самая большая доля площади у функции «{top_lu.index[0]}» ({top_lu.iloc[0]:.0f} %): это научный центр РАН и производственные зоны на востоке. Жилая функция занимает {lu_share.get('жилая', 0):.0f} %.",
    f"Население по модели — около {pop_total:,.0f} человек. В трёх самых населённых кварталах живёт {top3_pop:.0%} жителей: это многоэтажные микрорайоны в центре и на юго-западе. Оценка выше переписи 2021 г. (≈ 21 тыс.), потому что жилыми считаются все здания «yes» в жилых зонах, включая гаражи и сараи.".replace(",", " "),
    f"Пешком между центрами кварталов в среднем {summary['медианное время пешком между кварталами, мин']} мин (медиана). Самые «центральные» кварталы по доступности с учётом площади — около {blocks['area_accessibility'].min():.0f} мин, окраинные — до {blocks['area_accessibility'].max():.0f} мин.",
    f"Обеспеченность школами по модели: строгая {summary['provision_strong (город)']:.0%}, мягкая {summary['provision_weak (город)']:.0%}. Значения совпадают, потому что все {summary['мест в школах (оценка)']} мест заняты детьми из соседних кварталов (до 15 мин), и на дальние кварталы мест не остаётся. Вся вместимость — оценка BlocksNet (250 мест на школу): в OSM вместимость не указана, реальные школы крупнее. Поэтому нехватка мест, скорее всего, преувеличена.",
    "Застройка в целом разреженная: средний FSI застроенных кварталов {:.2f}. Spacematrix и «Стрелка» сходятся в главном: город состоит из малоэтажных нежилых кварталов (институты, промзоны) и нескольких средне- и многоэтажных жилых микрорайонов. Spacematrix делит их тоньше, «Стрелка» — по общей шкале для всех городов.".format(summary["средний FSI застроенных кварталов"]),
]
conclusions_html = "".join(f"<li>{html.escape(c)}</li>" for c in conclusions)

limitations = [
    "Кварталы нарезаны по дорогам OSM с тегами из pipeline.ipynb, поэтому они крупнее реальных: внутриквартальные проезды (service) не режут территорию.",
    "Пешеходный граф собран OSMnx из той же выгрузки OSM, время = длина / 5 км/ч. Рельеф, светофоры и закрытые территории не учитываются. Время считается между точками внутри кварталов, поэтому для больших кварталов оно завышено.",
    "Этажность есть в OSM только у 54 % зданий, остальным присвоен 1 этаж (оценка BlocksNet).",
    "Население = жилая площадь / 30 м² на человека: модельная оценка, а не статистика.",
    "Вместимость всех школ оценена (в OSM нет тега capacity). Колонка «в т. ч. оценённая вместимость» показывает, сколько мест — оценка.",
    "Данные OSM получены через API OpenStreetMap, потому что Overpass API во время работы был перегружен. Состав данных тот же.",
]
limitations_html = "".join(f"<li>{html.escape(c)}</li>" for c in limitations)

# ---------------------------------------------------------------- HTML

page = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Черноголовка: кварталы и метрики</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>__PLOTLYJS__</script>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f4f1; --surface-1: #fcfcfb; --border: #e2e1dc;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974; --accent: #2a78d6;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
  --empty: #e6e5e0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #33322f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e87; --accent: #3987e5;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
    --empty: #2b2b29;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #33322f;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e87; --accent: #3987e5;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  --empty: #2b2b29;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header, main { max-width: 1240px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 28px; padding-bottom: 8px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 0 0 8px; color: var(--text-secondary); font-weight: 600; }
.sub { color: var(--text-secondary); margin: 0; }
nav { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 12px; font-size: 14px; }
nav a { color: var(--accent); text-decoration: none; }
section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin: 16px 0; }
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
.stat { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.stat-v { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.stat-l { font-size: 13px; color: var(--text-secondary); }
.map-wrap { display: grid; grid-template-columns: 1fr 300px; gap: 14px; }
#map { height: 600px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-0); }
.panel label { font-size: 13px; color: var(--text-secondary); display: block; margin-bottom: 4px; }
select { width: 100%; padding: 7px; font: inherit; font-size: 14px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--text-primary); }
.legend { margin-top: 14px; font-size: 13px; }
.legend-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.sw { width: 18px; height: 12px; border-radius: 3px; flex: none; border: 1px solid var(--border); }
.note { font-size: 13px; color: var(--text-muted); margin-top: 10px; }
.badge { display: inline-block; font-size: 12px; padding: 1px 7px; border-radius: 10px; border: 1px solid var(--border); color: var(--text-secondary); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid2 > *, .map-wrap > * { min-width: 0; }
.chart { height: 380px; }
.tbl-wrap { overflow: auto; max-height: 460px; border: 1px solid var(--border); border-radius: 8px; }
table.tbl { border-collapse: collapse; font-size: 13px; width: 100%; font-variant-numeric: tabular-nums; }
.tbl th, .tbl td { padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
.tbl th { position: sticky; top: 0; background: var(--surface-1); color: var(--text-secondary); font-weight: 600; text-align: left; }
.tbl td:first-child, .tbl th:first-child { text-align: left; }
ul.list li { margin: 6px 0; }
.leaflet-container { font: inherit; }
.leaflet-popup-content { font-size: 12px; max-height: 300px; overflow: auto; }
.leaflet-popup-content table td { padding: 1px 6px 1px 0; }
footer { color: var(--text-muted); font-size: 13px; text-align: center; padding: 20px 16px 36px; }
@media (max-width: 860px) {
  .map-wrap, .grid2 { grid-template-columns: 1fr; }
  #map { height: 440px; }
}
</style>
</head>
<body>
<header>
  <h1>Черноголовка: кварталы и метрики BlocksNet</h1>
  <p class="sub">Город Черноголовка, Московская область, Россия · практическая работа №2 курса «Технологии ИИ в урбанистике» · BlocksNet 1.0.0a9</p>
  <nav><a href="#summary">Итоги</a><a href="#map-sec">Карта</a><a href="#charts">Графики</a><a href="#tables">Таблицы</a><a href="#conclusions">Выводы</a><a href="#sources">Источники и ограничения</a></nav>
</header>
<main>
<section id="summary"><h2>Главные цифры</h2><div class="stats">__SUMMARY__</div>
<p class="note">Население и вместимость школ — модельные оценки (см. «Ограничения»).</p></section>

<section id="map-sec"><h2>Интерактивная карта кварталов</h2>
<div class="map-wrap">
  <div id="map" role="img" aria-label="Карта кварталов Черноголовки с выбранным показателем"></div>
  <div class="panel">
    <label for="metric">Показатель на карте</label>
    <select id="metric"></select>
    <div id="metric-info" class="note"></div>
    <div id="legend" class="legend"></div>
    <div class="legend">
      <div class="legend-row"><span class="sw" style="background:#d93025;border-radius:50%;width:12px"></span>школа (точка)</div>
      <div class="legend-row"><span class="sw" style="height:3px;background:var(--text-secondary)"></span>связь «квартал → школа»</div>
    </div>
    <p class="note">Наведите курсор на квартал, чтобы увидеть значение. Нажмите на квартал, чтобы открыть все показатели. Серым показаны кварталы без значения. Шкала разбита на 5 равных по числу кварталов групп (квантили).</p>
  </div>
</div></section>

<section id="charts"><h2>Графики</h2>
<div class="grid2">
  <div><h3>Землепользование: площадь по преобладающей функции</h3><div id="ch-landuse" class="chart"></div></div>
  <div><h3>Здания: 15 самых населённых кварталов (оценка)</h3><div id="ch-population" class="chart"></div></div>
  <div><h3>Обеспеченность школами: как закрыт спрос (15 кварталов с наибольшим спросом)</h3><div id="ch-provision" class="chart"></div></div>
  <div><h3>Spacematrix: FSI против GSI, цвет — морфотип</h3><div id="ch-spacematrix" class="chart"></div></div>
  <div><h3>Типы застройки «Стрелка»: число кварталов</h3><div id="ch-strelka" class="chart"></div></div>
  <div><h3>Доступность с учётом площади: распределение кварталов</h3><div id="ch-area_acc" class="chart"></div></div>
  <div><h3>Транспорт: матрица времени пешком между кварталами, мин</h3><div id="ch-matrix" class="chart"></div></div>
  <div><h3>Пояснение к обеспеченности</h3>
    <p>Спрос — 120 мест на 1000 жителей, норматив — 15 минут пешком. <b>Строгая</b> обеспеченность — доля спроса, закрытая школами в пределах 15 минут, <b>мягкая</b> — закрытая вообще. <b>Нехватка</b> — места, которых не хватило. <b>Остаток</b> — свободные места в школах.</p>
    <p class="note">Все места в школах — оценка BlocksNet (по 250 на школу): в OSM вместимость не указана.</p>
  </div>
</div></section>

<section id="tables"><h2>Таблицы</h2>
<h3>Все показатели по кварталам</h3><div class="tbl-wrap">__BLOCKS_TABLE__</div>
<div class="grid2" style="margin-top:16px">
  <div><h3>Школы и вместимость</h3><div class="tbl-wrap">__SCHOOLS__</div>
       <h3 style="margin-top:14px">Связи «квартал спроса → квартал со школой»</h3><div class="tbl-wrap">__LINKS__</div></div>
  <div><h3>Характеристики групп Spacematrix</h3><div class="tbl-wrap">__CLUSTERS__</div></div>
</div>
<h3 style="margin-top:16px">Справочник показателей: единицы и происхождение</h3><div class="tbl-wrap">__GLOSSARY__</div>
</section>

<section id="conclusions"><h2>Выводы</h2><ul class="list">__CONCLUSIONS__</ul></section>

<section id="sources"><h2>Источники данных</h2><ul class="list">__SOURCES__</ul>
<p class="note">Данные © участники OpenStreetMap, лицензия ODbL. Подложка карты © участники OpenStreetMap (для неё и для библиотеки карты Leaflet нужен интернет; графики работают без интернета).</p>
<h2 style="margin-top:18px">Ограничения</h2><ul class="list">__LIMITATIONS__</ul></section>
</main>
<footer>Страница собрана скриптом make_dashboard.py из файлов results/*.geojson. Расчёты — в ноутбуке practice02_chernogolovka.ipynb.</footer>

<script>
const BLOCKS = __BLOCKS__;
const BOUNDARY = __BOUNDARY__;
const SCHOOLS = __SCHOOLS_PTS__;
const LINKS = __LINKS_GEO__;
const METRICS = __METRICS__;
const CHARTS = __CHARTS__;

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
const fmt = (v, unit) => v === null || v === undefined ? "нет данных"
  : (typeof v === "number" ? v.toLocaleString("ru-RU", {maximumFractionDigits: Math.abs(v) < 10 ? 2 : 0}) : v) + (unit && typeof v === "number" ? " " + unit : "");
const catColors = () => [1,2,3,4,5,6,7,8].map(i => css("--series-" + i));

// ---- карта
const map = L.map("map", {scrollWheelZoom: false});
map.fitBounds(L.geoJSON(BLOCKS).getBounds());  // сначала задаём вид, затем добавляем слои
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  {attribution: "© участники OpenStreetMap", maxZoom: 19, opacity: 0.6}).addTo(map);
L.geoJSON(BOUNDARY, {style: {color: "#d93025", weight: 2, fill: false}, interactive: false}).addTo(map);
const linkLayer = L.geoJSON(LINKS, {style: {color: css("--text-secondary"), weight: 2, dashArray: "4 4"},
  onEachFeature: (f, l) => l.bindTooltip(`кв. ${f.properties.from} → кв. ${f.properties.to}: ${fmt(f.properties.seats, "мест")}, ${fmt(f.properties.time_min, "мин")}`)});
let colorOf = () => css("--empty");
const blockLayer = L.geoJSON(BLOCKS, {
  style: f => ({color: "#ffffff", weight: 1, fillOpacity: 0.78, fillColor: colorOf(f.properties)}),
  onEachFeature: (f, l) => {
    l.on("mouseover", e => { e.target.setStyle({weight: 3, color: css("--text-primary")}); });
    l.on("mouseout", e => { blockLayer.resetStyle(e.target); });
    const rows = METRICS.map(m => `<tr><td>${m.label}</td><td><b>${fmt(f.properties[m.key], m.unit)}</b></td></tr>`).join("");
    l.bindPopup(`<b>Квартал ${f.properties.block_id}</b><table>${rows}</table>`);
  }
}).addTo(map);
linkLayer.addTo(map);
SCHOOLS.forEach(s => L.circleMarker([s.lat, s.lon], {radius: 7, color: "#ffffff", weight: 2, fillColor: "#d93025", fillOpacity: 1})
  .bindTooltip(`${s.name}<br>вместимость: ${fmt(s.capacity, "мест")} (${s.src})`).addTo(map));

function quantileBreaks(vals) {
  const s = [...vals].sort((a, b) => a - b), br = [];
  for (let i = 1; i < 5; i++) br.push(s[Math.floor(i * s.length / 5)]);
  return [...new Set(br)];
}
function setMetric(key) {
  const m = METRICS.find(x => x.key === key);
  const legend = document.getElementById("legend");
  document.getElementById("metric-info").innerHTML = `Группа: ${m.group}${m.unit ? " · единицы: " + m.unit : ""}<br><span class="badge">${m.origin}</span>`;
  const vals = BLOCKS.features.map(f => f.properties[key]);
  if (m.kind === "cat") {
    const counts = {};
    vals.forEach(v => { if (v !== null) counts[v] = (counts[v] || 0) + 1; });
    const cats = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    const cc = catColors(), map_ = {};
    cats.forEach((c, i) => map_[c] = i < 8 ? cc[i] : css("--text-muted"));
    if (key === "land_use_ru") map_["не определено"] = css("--empty");
    colorOf = p => p[key] === null ? css("--empty") : map_[p[key]];
    legend.innerHTML = cats.map(c => `<div class="legend-row"><span class="sw" style="background:${map_[c]}"></span>${c} (${counts[c]})</div>`).join("");
  } else {
    const num = vals.filter(v => v !== null && v > 0);
    const br = num.length ? quantileBreaks(num) : [];
    const cls = v => { let i = 0; while (i < br.length && v >= br[i]) i++; return i; };
    const steps = br.length + 1, pal = SEQ.slice(5 - steps);
    colorOf = p => (p[key] === null || p[key] <= 0) ? css("--empty") : pal[cls(p[key])];
    const edges = [Math.min(...num), ...br, Math.max(...num)];
    legend.innerHTML = num.length ? pal.map((c, i) => `<div class="legend-row"><span class="sw" style="background:${c}"></span>${fmt(edges[i], "")} – ${fmt(edges[i + 1], m.unit)}</div>`).join("")
      + `<div class="legend-row"><span class="sw" style="background:${css("--empty")}"></span>0 или нет данных</div>` : "нет ненулевых значений";
  }
  blockLayer.setStyle(f => ({fillColor: colorOf(f.properties)}));
  blockLayer.eachLayer(l => l.bindTooltip(`кв. ${l.feature.properties.block_id}: ${fmt(l.feature.properties[key], m.unit)}`, {sticky: true}));
}
const sel = document.getElementById("metric");
let lastGroup = null, og = null;
METRICS.forEach(m => {
  if (m.group !== lastGroup) { og = document.createElement("optgroup"); og.label = m.group; sel.appendChild(og); lastGroup = m.group; }
  const o = document.createElement("option"); o.value = m.key; o.textContent = m.label + (m.unit ? ` (${m.unit})` : ""); og.appendChild(o);
});
let saved = null; try { saved = localStorage.getItem("cg-metric"); } catch (e) {}
sel.value = METRICS.some(m => m.key === saved) ? saved : "bld_population";
sel.addEventListener("change", () => { setMetric(sel.value); try { localStorage.setItem("cg-metric", sel.value); } catch (e) {} });
setMetric(sel.value);

// ---- графики
function resolveColors(obj) {
  const cc = catColors();
  return JSON.parse(JSON.stringify(obj), (k, v) => {
    if (typeof v !== "string") return v;
    if (/^series\\d$/.test(v)) return css("--" + v.replace("series", "series-"));
    if (/^cat\\d+$/.test(v)) { const i = +v.slice(3); return i < 8 ? cc[i] : css("--text-muted"); }
    if (v === "surface") return css("--surface-1");
    if (v === "seq") return SEQ.map((c, i) => [i / (SEQ.length - 1), c]);
    return v;
  });
}
function drawCharts() {
  const base = {paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {family: "system-ui, Segoe UI, sans-serif", size: 12, color: css("--text-secondary")},
    margin: {l: 60, r: 16, t: 10, b: 60}, hoverlabel: {font: {size: 13}},
    xaxis: {gridcolor: css("--border"), zerolinecolor: css("--border"), automargin: true},
    yaxis: {gridcolor: css("--border"), zerolinecolor: css("--border"), automargin: true}};
  Object.entries(CHARTS).forEach(([id, ch]) => {
    const c = resolveColors(ch);
    const layout = Object.assign({}, base, c.layout,
      {xaxis: Object.assign({}, base.xaxis, c.layout.xaxis || {}), yaxis: Object.assign({}, base.yaxis, c.layout.yaxis || {}),
       margin: Object.assign({}, base.margin, c.layout.margin || {})});
    Plotly.react("ch-" + id, c.data, layout, {displayModeBar: false, responsive: true});
  });
}
drawCharts();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawCharts(); setMetric(sel.value); });
</script>
</body>
</html>
"""

replacements = {
    "__SUMMARY__": summary_rows,
    "__BLOCKS_TABLE__": blocks_table_html,
    "__SCHOOLS__": schools_html,
    "__LINKS__": links_html,
    "__CLUSTERS__": clusters_html,
    "__GLOSSARY__": glossary_html,
    "__CONCLUSIONS__": conclusions_html,
    "__SOURCES__": sources_html,
    "__LIMITATIONS__": limitations_html,
    "__BLOCKS__": json.dumps(blocks_geojson, ensure_ascii=False),
    "__BOUNDARY__": json.dumps(boundary_geojson, ensure_ascii=False),
    "__SCHOOLS_PTS__": json.dumps(schools_points, ensure_ascii=False),
    "__LINKS_GEO__": json.dumps(links_geojson, ensure_ascii=False),
    "__METRICS__": json.dumps(metrics_js, ensure_ascii=False),
    "__CHARTS__": json.dumps(charts, ensure_ascii=False),
}
# Библиотека графиков Plotly встраивается в страницу целиком (из установленного пакета plotly),
# поэтому графики работают без интернета. Интернет нужен только для подложки карты и Leaflet.
from plotly.offline import get_plotlyjs

replacements["__PLOTLYJS__"] = get_plotlyjs()
for key, value in replacements.items():
    page = page.replace(key, value)

OUT_FILE.write_text(page, encoding="utf-8")
print(f"Готово: {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.0f} КБ). Откройте файл двойным щелчком в браузере.")
