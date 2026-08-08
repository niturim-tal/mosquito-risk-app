# ==============================================================================
# מודל מרחבי לחיזוי וניטור סיכון יתושים - אפליקציית Streamlit
# ==============================================================================

import streamlit as st
import ee
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import geemap
import folium
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. הגדרות תצוגת הדף ואתחול Google Earth Engine באמצעות Service Account
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="מודל סיכון יתושים מרחבי",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_earth_engine():
    """
    מאתחלת את Google Earth Engine באמצעות Service Account מתוך ה-Secrets של Streamlit.
    """
    try:
        if "GEE_SERVICE_ACCOUNT_KEY" in st.secrets:
            # הפיכת ה-Secrets למילון Python תקין
            key_dict = dict(st.secrets["GEE_SERVICE_ACCOUNT_KEY"])
            
            # ניקוי תווי שורה חדשה במידה והוזנו כתו מילולי
            if "private_key" in key_dict:
                key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")

            # התחברות ל-Earth Engine בעזרת המילון
            credentials = ee.ServiceAccountCredentials(
                key_dict['client_email'],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials, project="hybrid-dolphin-318308")
            return True
        else:
            st.error("⚠️ ה-Secret בשם 'GEE_SERVICE_ACCOUNT_KEY' לא נמצא בהגדרות ה-Secrets.")
            return False
            
    except Exception as e:
        st.error(f"❌ שגיאה באימות מול Google Earth Engine: {e}")
        return False

if not init_earth_engine():
    st.stop()


# ------------------------------------------------------------------------------
# 1. פונקציות עזר כלליות ואקלים
# ------------------------------------------------------------------------------

def reverse_hebrew(text):
    """מחזירה טקסט בסדר הפוך לפתרון הצגת עברית ב-Matplotlib."""
    if not text:
        return text
    return text[::-1]


def get_era5_climatology(month, day):
    """שולפת ממוצע אקלימי היסטורי מ-ERA5-Land."""
    era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    era5_recent = era5_land.filter(ee.Filter.calendarRange(2020, 2024, 'year'))

    day_start = max(1, day - 3)
    day_end = min(31, day + 3)

    temp_coll = era5_recent.filter(
        ee.Filter.And(
            ee.Filter.calendarRange(month, month, 'month'),
            ee.Filter.calendarRange(day_start, day_end, 'day_of_month')
        )
    ).select('temperature_2m')
    avg_temp = temp_coll.mean().subtract(273.15).rename('temperature_2m_above_ground')

    rain_coll = era5_recent.select('total_precipitation_sum')
    rain_current_coll = rain_coll.filter(
        ee.Filter.And(
            ee.Filter.calendarRange(month, month, 'month'),
            ee.Filter.calendarRange(day_start, day_end, 'day_of_month')
        )
    )

    rain_current = (
        rain_current_coll.mean()
        .multiply(1000)
        .gte(1.5)
        .rename('current_rain')
    )

    prev_day_start = max(1, day - 21)
    prev_day_end = max(1, day - 7)

    if day <= 21:
        prev_month = month - 1 if month > 1 else 12
        rain_previous_coll = rain_coll.filter(
            ee.Filter.And(
                ee.Filter.calendarRange(prev_month, month, 'month'),
                ee.Filter.calendarRange(1, 31, 'day_of_month')
            )
        )
    else:
        rain_previous_coll = rain_coll.filter(
            ee.Filter.And(
                ee.Filter.calendarRange(month, month, 'month'),
                ee.Filter.calendarRange(prev_day_start, prev_day_end, 'day_of_month')
            )
        )

    rain_previous = (
        rain_previous_coll.mean()
        .multiply(1000)
        .gte(1.5)
        .rename('previous_rain')
    )

    return avg_temp, rain_current, rain_previous


# ------------------------------------------------------------------------------
# 2. ניתוח סדרות זמן חודשיות
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def calculate_annual_mosquito_risk(selected_year, _geometry_to_clip):
    """מחשבת סדרת זמן של סיכון יתושים חודשי בערכי השרת של Earth Engine."""
    months = ee.List.sequence(1, 12)

    def calculate_month_mean(m):
        m = ee.Number(m)
        month_start = ee.Date.fromYMD(selected_year, m, 1)
        days_sample_seq = ee.List.sequence(0, 28, 3)

        def calculate_daily_risk(d):
            day_date = month_start.advance(ee.Number(d), 'day')
            month_num = day_date.get('month')
            day_num = day_date.get('day')

            era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filter(ee.Filter.calendarRange(2020, 2024, 'year'))
            t_clim = era5_land.filter(ee.Filter.And(
                ee.Filter.calendarRange(month_num, month_num, 'month'),
                ee.Filter.calendarRange(ee.Number(day_num).subtract(3).max(1), ee.Number(day_num).add(3).min(31), 'day_of_month')
            )).select('temperature_2m').mean().subtract(273.15)

            c_rain_clim = era5_land.filter(ee.Filter.And(
                ee.Filter.calendarRange(month_num, month_num, 'month'),
                ee.Filter.calendarRange(ee.Number(day_num).subtract(3).max(1), ee.Number(day_num).add(3).min(31), 'day_of_month')
            )).select('total_precipitation_sum').mean().multiply(1000).gte(1.5)

            p_rain_clim = era5_land.filter(ee.Filter.calendarRange(month_num, month_num, 'month')) \
                .select('total_precipitation_sum').mean().multiply(1000).gte(1.5)

            gfs_coll = ee.ImageCollection('NOAA/GFS0P25') \
                .filter(ee.Filter.date(day_date, day_date.advance(1, 'day'))) \
                .filter(ee.Filter.eq('forecast_hours', 3))

            has_gfs = gfs_coll.size().gt(0)

            t_img = ee.Image(ee.Algorithms.If(has_gfs, gfs_coll.select('temperature_2m_above_ground').mean(), t_clim))
            c_rain = ee.Image(ee.Algorithms.If(has_gfs, gfs_coll.select('total_precipitation_surface').map(lambda img: img.gte(5)).max(), c_rain_clim))
            p_rain = p_rain_clim

            risk_img = ee.Image.constant(0).expression(
                "((C_rain == 0) && ((T >= 20 && T < 23) || (T >= 15 && P_rain == 1))) ? 4 : " +
                "((C_rain == 0) && (T >= 15 && T < 20)) ? 3 : " +
                "((C_rain == 0) && ((T >= 14 && T < 15) || T >= 23)) ? 2 : " +
                "1",
                {'T': t_img, 'C_rain': c_rain, 'P_rain': p_rain}
            )

            daily_city_mean = risk_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=_geometry_to_clip.geometry(),
                scale=1000,
                maxPixels=1e9,
                tileScale=4
            ).get('constant')

            return daily_city_mean

        daily_means = days_sample_seq.map(calculate_daily_risk)
        valid_daily_means = daily_means.removeAll([None])
        monthly_avg = ee.Number(valid_daily_means.reduce(ee.Reducer.mean()))

        return ee.Feature(None, {'month': m, 'monthly_mosquito_risk': monthly_avg})

    monthly_features = ee.FeatureCollection(months.map(calculate_month_mean))
    res_dict = monthly_features.reduceColumns(
        ee.Reducer.toList().repeat(2), ['month', 'monthly_mosquito_risk']
    ).getInfo()

    return dict(zip(res_dict['list'][0], res_dict['list'][1]))


# ------------------------------------------------------------------------------
# 3. הפקת הגרף ב-Matplotlib
# ------------------------------------------------------------------------------

def create_monthly_chart(monthly_data, city_name, year):
    """מייצרת אובייקט matplotlib.figure להצגה ב-Streamlit."""
    df = pd.DataFrame({
        'Month': [str(m) for m in range(1, 13)],
        'Risk_Level': [monthly_data.get(m, 0) for m in range(1, 13)]
    })

    colors = []
    for v in df['Risk_Level']:
        if v >= 3.5: colors.append('#FF0000')
        elif v >= 2.5: colors.append('#FFA500')
        elif v >= 1.5: colors.append('#FFFF00')
        else: colors.append('#00FF00')

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    bars = ax.bar(df['Month'], df['Risk_Level'], color=colors, edgecolor='black', linewidth=0.5)

    ax.set_ylim(0, 4.5)
    ax.set_xlabel(reverse_hebrew('חודש'), fontsize=10, fontweight='bold')
    ax.set_ylabel(reverse_hebrew('רמת סיכון ממוצעת'), fontsize=10, fontweight='bold')
    title_text = f"({year}) {city_name} - {reverse_hebrew('סיכון יתושים חודשי')}"
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        if height and height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# 4. טעינת רשימת הערים מ-GEE
# ------------------------------------------------------------------------------

@st.cache_data
def get_city_options():
    try:
        cities_collection = ee.FeatureCollection('projects/hybrid-dolphin-318308/assets/simplified')
        city_list = cities_collection.aggregate_array('Muni_Eng').getInfo()
        return sorted(list(set([c for c in city_list if c])))
    except Exception as e:
        st.error(f"⚠️ לא ניתן שטוף רשימת ערים מה-Asset: {e}")
        return []

city_options = get_city_options()


# ------------------------------------------------------------------------------
# 5. ממשק המשתמש (UI Layout)
# ------------------------------------------------------------------------------

# כותרת ראשית
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-right: 5px solid #007bff; text-align: center; margin-bottom: 20px;">
        <h2 style="margin:0; color: #111;">🦟 מודל סיכון יתושים מרחבי</h2>
        <p style="margin:0; color: #666; font-size: 14px;">ניטור וחיזוי מבוסס נתונים מטאורולוגיים מבוססי ענן (NOAA GFS & ERA5-Land)</p>
    </div>
""", unsafe_allow_html=True)


# פאנל שליטה בסרגל הצדי (Sidebar)
with st.sidebar:
    st.header("⚙️ פרמטרים לניתוח")
    
    target_date = st.date_input(
        "📅 בחר תאריך:",
        value=datetime(2026, 8, 15).date()
    )
    
    selected_city = st.selectbox(
        "📍 בחר יישוב:",
        options=city_options,
        index=city_options.index("Shoham") if "Shoham" in city_options else 0
    )
    
    run_analysis = st.button("🚀 הרץ ניתוח מרחבי", type="primary", use_container_width=True)


# ------------------------------------------------------------------------------
# 6. הליבה - עיבוד והצגה
# ------------------------------------------------------------------------------

if run_analysis or selected_city:
    with st.spinner(f"מחשב נתונים עבור {selected_city}..."):
        target_date_str = target_date.strftime('%Y-%m-%d')
        target_date_obj = datetime.strptime(target_date_str, '%Y-%m-%d')
        today_obj = datetime.now()
        selected_year = target_date_obj.year

        countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        israel_extended = countries.filter(ee.Filter.inList('country_na', ['Israel', 'West Bank', 'Gaza Strip']))
        israel_mask = ee.Image.constant(1).clip(israel_extended)

        cities_collection = ee.FeatureCollection('projects/hybrid-dolphin-318308/assets/simplified')
        city_feature = cities_collection.filter(ee.Filter.eq('Muni_Eng', selected_city))

        if city_feature.size().getInfo() == 0:
            st.error(f"⚠️ היישוב '{selected_city}' לא נמצא במאגר.")
            st.stop()

        geometry_to_clip = city_feature
        days_ahead = (target_date_obj.date() - today_obj.date()).days

        end_date_current_obj = target_date_obj - timedelta(days=7)
        end_date_current_str = end_date_current_obj.strftime('%Y-%m-%d')
        start_date_previous_str = end_date_current_str
        end_date_previous_obj = target_date_obj - timedelta(days=21)
        end_date_previous_str = end_date_previous_obj.strftime('%Y-%m-%d')

        if days_ahead <= 7:
            mode_desc = "NOAA GFS"
            gfs_current = ee.ImageCollection('NOAA/GFS0P25').filter(ee.Filter.date(end_date_current_str, target_date_str)).filter(ee.Filter.eq('forecast_hours', 3))
            gfs_previous = ee.ImageCollection('NOAA/GFS0P25').filter(ee.Filter.date(end_date_previous_str, start_date_previous_str)).filter(ee.Filter.eq('forecast_hours', 3))

            weekly_mean_temp = gfs_current.select('temperature_2m_above_ground').mean()
            rain_event_current = gfs_current.select('total_precipitation_surface').map(lambda img: img.gte(5)).max().rename('current_rain')
            rain_event_previous = gfs_previous.select('total_precipitation_surface').map(lambda img: img.gte(5)).max().rename('previous_rain')

        elif 7 < days_ahead <= 21:
            mode_desc = "Hybrid (GFS/ERA5)"
            weekly_mean_temp, rain_event_current, _ = get_era5_climatology(target_date_obj.month, target_date_obj.day)
            gfs_previous = ee.ImageCollection('NOAA/GFS0P25').filter(ee.Filter.date(end_date_previous_str, start_date_previous_str)).filter(ee.Filter.eq('forecast_hours', 3))
            rain_event_previous = gfs_previous.select('total_precipitation_surface').map(lambda img: img.gte(5)).max().rename('previous_rain')

        else:
            mode_desc = "ERA5 Climatology"
            weekly_mean_temp, rain_event_current, rain_event_previous = get_era5_climatology(target_date_obj.month, target_date_obj.day)

        weekly_mean_temp = weekly_mean_temp.updateMask(israel_mask)
        rain_event_current = rain_event_current.updateMask(israel_mask)
        rain_event_previous = rain_event_previous.updateMask(israel_mask)

        mosquito_risk = ee.Image.constant(0).expression(
            "((C_rain == 0) && ((T >= 20 && T < 23) || (T >= 15 && P_rain == 1))) ? 4 : " +
            "((C_rain == 0) && (T >= 15 && T < 20)) ? 3 : " +
            "((C_rain == 0) && ((T >= 14 && T < 15) || T >= 23)) ? 2 : " +
            "1",
            {'T': weekly_mean_temp, 'C_rain': rain_event_current, 'P_rain': rain_event_previous}
        ).rename('mosquito_risk_index').updateMask(israel_mask)

        mosquito_risk_clipped = mosquito_risk.clip(geometry_to_clip)
        weekly_mean_temp_clipped = weekly_mean_temp.clip(geometry_to_clip)

        # חילוץ מדדים
        mean_temp_val = weekly_mean_temp_clipped.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry_to_clip.geometry(), scale=250
        ).get('temperature_2m_above_ground').getInfo() or 0.0

        rain_curr_val = rain_event_current.clip(geometry_to_clip).reduceRegion(
            reducer=ee.Reducer.max(), geometry=geometry_to_clip.geometry(), scale=250
        ).get('current_rain').getInfo() or 0

        rain_prev_val = rain_event_previous.clip(geometry_to_clip).reduceRegion(
            reducer=ee.Reducer.max(), geometry=geometry_to_clip.geometry(), scale=250
        ).get('previous_rain').getInfo() or 0

        risk_val = mosquito_risk_clipped.reduceRegion(
            reducer=ee.Reducer.max(), geometry=geometry_to_clip.geometry(), scale=250
        ).get('mosquito_risk_index').getInfo() or 1

        # חלוקת המסך לתצוגה (2 עמודות)
        col1, col2 = st.columns([2, 1])

with col1:
            st.subheader("🗺️ תצוגה מרחבית (GEE Map)")
            
            # יצירת מפת Folium נקייה
            m = folium.Map(location=[31.0461, 34.8516], zoom_start=12)

            risk_vis = {'min': 1, 'max': 4, 'palette': ['#00FF00', '#FFFF00', '#FFA500', '#FF0000']}
            temp_vis = {'min': 14.0, 'max': 38.0, 'palette': ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']}

            # חילוץ שכבת הסיכון מ-GEE והוספתה למפה
            risk_map_id = ee.Image(mosquito_risk_clipped).getMapId(risk_vis)
            folium.TileLayer(
                tiles=risk_map_id['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='Mosquito Risk Level (1-4)',
                overlay=True,
                control=True
            ).add_to(m)

            # חילוץ שכבת הטמפרטורה מ-GEE והוספתה למפה
            temp_map_id = ee.Image(weekly_mean_temp_clipped).getMapId(temp_vis)
            folium.TileLayer(
                tiles=temp_map_id['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name='Weekly Mean Temp (C)',
                overlay=True,
                control=True
            ).add_to(m)

            # הוספת פקד שליטה בשכבות
            folium.LayerControl().add_to(m)

            # רינדור המפה ב-Streamlit
            st_folium(m, width="100%", height=500)            
            legend_keys = ['Low Risk (1)', 'Moderate Risk (2)', 'High Risk (3)', 'Critical Risk (4)']
            legend_colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000']
            m.add_legend(title="Mosquito Risk Legend", keys=legend_keys, colors=legend_colors)

            # רינדור המפה ב-Streamlit
            st_folium(m, width="100%", height=500)

        with col2:
            st.subheader("📋 נתוני סיכום")
            
            risk_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
            risk_names = {1: "נמוך", 2: "בינוני", 3: "גבוה", 4: "קריטי"}
            
            st.metric(
                label="רמת סיכון מחושבת",
                value=f"{risk_colors.get(int(risk_val), '')} {int(risk_val)} / 4 ({risk_names.get(int(risk_val), '')})"
            )
            
            st.markdown(f"""
            * 📍 **יישוב:** {selected_city}
            * 📅 **תאריך:** {target_date_str}
            * 🌡️ **טמפרטורה ממוצעת:** `{mean_temp_val:.1f}°C`
            * 🌧️ **אירוע גשם נוכחי:** `{'כן' if rain_curr_val else 'לא'}`
            * 🌧️ **אירוע גשם מקדים:** `{'כן' if rain_prev_val else 'לא'}`
            * 📡 **מקור נתונים:** `{mode_desc}`
            """)

            st.divider()
            st.subheader("📊 ניתוח חודשי שנתי")
            monthly_data = calculate_annual_mosquito_risk(selected_year, geometry_to_clip)
            fig = create_monthly_chart(monthly_data, selected_city, selected_year)
            st.pyplot(fig)
