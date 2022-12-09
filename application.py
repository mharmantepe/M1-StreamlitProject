### MELIS HARMANTEPE ###
from PIL import Image
from functools import wraps
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)
# Misc logger setup so a debug log statement gets printed on stdout
logger.setLevel("DEBUG")
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)


def timeit(func):
    # This decorator prints the execution time for the decorated function
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} started at {} and ran in {}seconds".format(
            func.__name__, start, round(end - start, 2)))
        return result
    return wrapper


st.title('Real Estate Sales Prices')

DATE_COLUMN = 'date_mutation'
FILE_2020 = 'df_2020_sampled_small.csv'
FILE_2019 = 'df_2019_sampled_small.csv'
FILE_2018 = 'df_2018_sampled_small.csv'
FILE_2017 = 'df_2017_sampled_small.csv'
BACKGROUND = "background.png"

#FILE_2020 = 'full_2020.csv'
#FILE_2019 = 'full_2019.csv'
#FILE_2018 = 'full_2018.csv'
#FILE_2017 = 'full_2017.csv'


@timeit
@st.cache
def load_data(FILE):
    data = pd.read_csv(FILE)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data = data[["id_parcelle", "date_mutation", "nature_mutation", "valeur_fonciere", "code_postal", "code_commune", "nom_commune",
                 "code_departement", "code_type_local", "type_local", "nombre_pieces_principales", "surface_reelle_bati", "nombre_lots", "longitude", "latitude"]].dropna()
    return data


def toint(x):
    try:
        return int(x)
    except Exception:
        return x


def get_year(dt):
    return dt.year


def count_rows(rows):
    return len(rows)


def preprocessing(data):
    toint(data["code_commune"])
    toint(data["code_departement"])
    toint(data["code_postal"])
    # Filter to keep only sales
    mask = (data["nature_mutation"] == "Vente") & (data["nature_mutation"] ==
                                                   "Vente en l'état futur d'achèvement") & (data["nature_mutation"] == "Vente terrain à bâtir")
    data = data[mask]
    return data


def merge_datasets(data1, data2, data3, data4):
    union = pd.concat([data1, data2, data3, data4])
    return union


def metrics(data1, data2):
    st.subheader("Sales Metrics 🏙️")
    nb_sold_a = data1["id_parcelle"].map(count_rows)
    nb_sold_b = data2["id_parcelle"].map(count_rows)
    avg_price_a = (data1["valeur_fonciere"] /
                   data1["surface_reelle_bati"]).mean()
    avg_price_b = (data2["valeur_fonciere"] /
                   data2["surface_reelle_bati"]).mean()
    sold = len(nb_sold_a) - len(nb_sold_b)
    price = int(avg_price_a - avg_price_b)
    col1, col2 = st.columns(2)
    col1.metric("Number of Apartments Sold",
                value=len(nb_sold_a), delta=sold)
    col2.metric("Average Price per Square Meter",
                value=int(avg_price_a), delta=price)


def price_evolution_by_year(data):
    town_name, nb_room = st.columns(2)
    with town_name:
        nom_commune = st.selectbox(
            'Select a town', options=data["nom_commune"].values)
    mask = (data["nom_commune"] == nom_commune)
    data_new = data[mask]

    data_new["Year"] = data_new[DATE_COLUMN].map(get_year)
    by_year = data_new.groupby(data_new["Year"]).mean("valeur_fonciere")
    by_year = by_year.reset_index()

    fig = px.line(by_year, x="Year", y="valeur_fonciere")
    st.write('Average Price of Sales In %s Over the Years' % nom_commune)
    fig.update_xaxes(title_text='Price', dtick=1)
    fig.update_yaxes(title_text='Years')
    st.plotly_chart(fig, use_container_width=True)


def price_by_type_over_years(data):
    data["Year"] = data[DATE_COLUMN].map(get_year)
    by_type = data.groupby(["type_local", "Year"]).mean("valeur_fonciere")
    by_type = by_type.reset_index()
    x = ['2020', '2019', '2018', '2017']

    plot = go.Figure(data=[go.Bar(
        name='Maison',
        x=x,
        y=(by_type["valeur_fonciere"].where(
            by_type['type_local'] == 'Maison')).dropna().tolist()
    ),
        go.Bar(
        name='Local industriel/commercial/assimilé',
        x=x,
        y=by_type["valeur_fonciere"].where(by_type['type_local']
                                           == 'Local industriel. commercial ou assimilé').dropna().tolist()
    ),
        go.Bar(
        name='Appartement',
        x=x,
        y=(by_type["valeur_fonciere"].where(
            by_type['type_local'] == 'Appartement')).dropna().tolist()
    ),
        go.Bar(
            name='Dépendance',
            x=x,
            y=(by_type["valeur_fonciere"].where(
                by_type['type_local'] == 'Dépendance')).fillna(0)[1:5].tolist()
    )
    ])

    plot.update_layout(title_text="Price Evolution by Type Over the Years")
    plot.update_xaxes(title_text='Year', dtick=1)
    plot.update_yaxes(title_text='Price')
    st.plotly_chart(plot, use_container_width=True)


def nb_rooms(data):
    option_nb_piece = st.selectbox('How many rooms ?',
                                   data['nombre_pieces_principales'].sort_values().unique())
    mask_nb_piece = data['nombre_pieces_principales'] == option_nb_piece
    return mask_nb_piece


def raw_data_checkbox(data):
    a = st.empty()
    if a.checkbox('Show the list of all sales', key="Sales"):
        st.subheader('All Sales')
        nb = nb_rooms(data)
        st.dataframe(data[nb])


def nb_sales_by_date_hist(data):
    st.subheader('Number of Sales by Date')
    fig, ax = plt.subplots()
    ax.hist(data[DATE_COLUMN], bins=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    st.plotly_chart(fig, use_container_width=True)


def sales_by_department_and_surface(data):
    st.subheader('Price of Sales by Department')
    by_department = data.groupby(
        data["code_departement"]).mean("valeur_fonciere")
    by_department = by_department.reset_index()

    fig = px.bar(by_department, x="code_departement", y="valeur_fonciere",
                 color='surface_reelle_bati', barmode='group',
                 height=400)
    st.plotly_chart(fig, use_container_width=True)


def cheap_exp_checkbox(data):
    st.subheader('The Most Expensive and The Cheapest Sold Towns')
    b = st.empty()
    if b.checkbox('Click to see the cheapest towns'):
        st.subheader('Top 10 Cheapest Towns')
        top_10_cheapest_towns(data)


def top_10_most_expensive_towns(data):
    st.subheader('Top 10 Most Expensive Towns')
    by_town = data.groupby(data["nom_commune"]).mean('valeur_fonciere').sort_values(
        'valeur_fonciere', ascending=False)[:10]
    by_town = by_town.reset_index()
    st.bar_chart(data=by_town, x="nom_commune", y="valeur_fonciere")


def top_10_cheapest_towns(data):
    by_town = data.groupby(data["nom_commune"]).mean('valeur_fonciere').sort_values(
        'valeur_fonciere', ascending=False).tail(10)
    by_town = by_town.reset_index()
    st.bar_chart(data=by_town, x="nom_commune", y="valeur_fonciere")


def pie_chart_frequency(data):
    by_type = data.groupby(["type_local"]).apply(count_rows)
    by_type = by_type.reset_index()
    fig = px.pie(by_type, names="type_local", values=0)
    st.plotly_chart(fig)


def biggest_cities(data):
    st.header("Real Estate Price in the Top 10 Biggest Cities")
    cities = st.multiselect(
        "Choose a city",
        ("Paris", "Marseille", "Toulouse", "Lyon", "Nice",
         "Nantes", "Strasbourg", "Bordeaux", "Montpellier", "Rouen")
    )
    by_department = data.groupby(
        data["nom_commune"]).mean("valeur_fonciere")
    by_department = by_department.reset_index()
    plot = by_department["nom_commune"].isin(cities)
    fig = px.bar(by_department[plot], x="nom_commune", y="valeur_fonciere", barmode='group',
                 height=400)
    st.plotly_chart(fig, use_container_width=True)


def table_city_nb_rooms(data):
    st.header("All Cities and Number of Rooms")
    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input(
            "Enter a city name 👇"
        )
    with col2:
        number = st.number_input('Choose a room number')
        mask2 = data["nombre_pieces_principales"] == number
    if city:
        mask1 = data["nom_commune"] == city
        data = data[mask1]
        data = data[mask2]
    data = data[mask2]
    st.dataframe(data)


def tabs(data):
    tab1, tab2, tab3 = st.tabs(
        ["📈 Departments", "🗃 Frequency", "🗃 Correlations"])
    with tab1:
        sales_by_department_and_surface(data)
    with tab2:
        nb_sales_by_date_hist(data)
    with tab3:
        interactiveplot(data)


def interactiveplot(data):
    st.subheader('Check the correlation of attributes')
    col1, col2 = st.columns(2)

    x_axis_val = col1.selectbox('Select the X-axis', options=data.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=data.columns)

    plot = px.scatter(data, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=False)


def sales_by_towns_map(data):
    st.subheader('Map of Sales')
    min, max = st.slider('Select a price range',
                         value=[0.0, 800000.0])
    st.write('Selected Price Range:', min, max)
    month_to_filter = st.select_slider(
        'Select a month',
        options=['January', 'February', 'March', "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    st.write('All sales in %s' % month_to_filter)
    if month_to_filter == "January":
        filtered_data = data[data[DATE_COLUMN].dt.month == 1]
    elif month_to_filter == "February":
        filtered_data = data[data[DATE_COLUMN].dt.month == 2]
    elif month_to_filter == "March":
        filtered_data = data[data[DATE_COLUMN].dt.month == 3]
    elif month_to_filter == "April":
        filtered_data = data[data[DATE_COLUMN].dt.month == 4]
    elif month_to_filter == "May":
        filtered_data = data[data[DATE_COLUMN].dt.month == 5]
    elif month_to_filter == "June":
        filtered_data = data[data[DATE_COLUMN].dt.month == 6]
    elif month_to_filter == "July":
        filtered_data = data[data[DATE_COLUMN].dt.month == 7]
    elif month_to_filter == "August":
        filtered_data = data[data[DATE_COLUMN].dt.month == 8]
    elif month_to_filter == "September":
        filtered_data = data[data[DATE_COLUMN].dt.month == 9]
    elif month_to_filter == "October":
        filtered_data = data[data[DATE_COLUMN].dt.month == 10]
    elif month_to_filter == "November":
        filtered_data = data[data[DATE_COLUMN].dt.month == 11]
    else:
        filtered_data = data[data[DATE_COLUMN].dt.month == 12]

    price_filtered = filtered_data[(
        filtered_data["valeur_fonciere"] >= min) & (filtered_data["valeur_fonciere"] <= max)]
    st.map(price_filtered)


def main():
    image = Image.open('background.png')

    # Sidebar navigation
    with st.sidebar:
        options = option_menu(menu_title='Menu:', options=[
            'General View', '2020 Analysis📈', '2019 Analysis📈', '2018 Analysis📈', '2017 Analysis📈', ])

    # Navigation options
    if options == 'General View':
        st.image(image, caption='Real Estate in France Between 2017 and 2020')
        st.write('Welcome to this page where you can have an overview of the real estate sales in France between the years 2017 and 2020. If you want to see analysis of real estate data, you are at the right page!')
        data_load_state = st.text('Loading data...')
        data_2020 = load_data(FILE_2020)
        data_load_state.text(
            "Data from year 2020 has been loaded successfuly!")
        data_load_state = st.text('Loading data...')
        data_2019 = load_data(FILE_2019)
        data_load_state.text(
            "Data from year 2019 has been loaded successfuly!")
        data_load_state = st.text('Loading data...')
        data_2018 = load_data(FILE_2018)
        data_load_state.text(
            "Data from year 2018 has been loaded successfuly!")
        data_load_state = st.text('Loading data...')
        data_2017 = load_data(FILE_2017)
        data_load_state.text(
            "Data from year 2017 has been loaded successfuly!")
        sales = merge_datasets(data_2020, data_2019, data_2018, data_2017)
        preprocessing(sales)
        raw_data_checkbox(sales)
        price_evolution_by_year(sales)
        price_by_type_over_years(sales)

    elif options == '2020 Analysis📈':
        data_load_state = st.text('Loading data...')
        data_2020 = load_data(FILE_2020)
        data_load_state.text(
            "Data from year 2020 has been loaded successfuly!")
        data_2019 = load_data(FILE_2019)
        preprocessing(data_2019)
        preprocessing(data_2020)
        metrics(data_2020, data_2019)
        pie_chart_frequency(data_2020)
        biggest_cities(data_2020)
        table_city_nb_rooms(data_2020)
        sales_by_towns_map(data_2020)
        cheap_exp_checkbox(data_2020)
        top_10_most_expensive_towns(data_2020)
        tabs(data_2020)

    elif options == '2019 Analysis📈':
        data_load_state = st.text('Loading data...')
        data_2019 = load_data(FILE_2019)
        data_load_state.text(
            "Data from year 2019 has been loaded successfuly!")
        data_2018 = load_data(FILE_2018)
        preprocessing(data_2019)
        preprocessing(data_2018)
        metrics(data_2019, data_2018)
        pie_chart_frequency(data_2019)
        biggest_cities(data_2019)
        table_city_nb_rooms(data_2019)
        sales_by_towns_map(data_2019)
        cheap_exp_checkbox(data_2019)
        top_10_most_expensive_towns(data_2019)
        tabs(data_2019)

    elif options == '2018 Analysis📈':
        data_load_state = st.text('Loading data...')
        data_2018 = load_data(FILE_2018)
        data_load_state.text(
            "Data from year 2018 has been loaded successfuly!")
        data_2017 = load_data(FILE_2017)
        preprocessing(data_2018)
        preprocessing(data_2017)
        metrics(data_2018, data_2017)
        pie_chart_frequency(data_2018)
        biggest_cities(data_2018)
        table_city_nb_rooms(data_2018)
        sales_by_towns_map(data_2018)
        cheap_exp_checkbox(data_2018)
        top_10_most_expensive_towns(data_2018)
        tabs(data_2018)

    elif options == '2017 Analysis📈':
        data_load_state = st.text('Loading data...')
        data_2017 = load_data(FILE_2017)
        data_load_state.text(
            "Data from year 2017 has been loaded successfuly!")
        preprocessing(data_2017)
        pie_chart_frequency(data_2017)
        biggest_cities(data_2017)
        sales_by_towns_map(data_2017)
        nb_sales_by_date_hist(data_2017)
        sales_by_department_and_surface(data_2017)
        cheap_exp_checkbox(data_2017)
        top_10_most_expensive_towns(data_2017)
        tabs(data_2017)


if __name__ == "__main__":
    main()
