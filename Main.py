import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pickle.load(open('House_Price_Prediction.sav', 'rb'))
scaler = pickle.load(open('scaler_vf.pkl', 'rb'))  

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

def reset_form():
    st.session_state['submitted'] = False

st.title("🏠 Madrid House Price Prediction App")
st.write("---")  
st.info("💡 Utilisez cet outil pour estimer le prix d'une maison à Madrid. Notez que ce n'est qu'une estimation basée sur le modèle, et non un prix définitif.")

if not st.session_state['submitted']:
    sq_mt_built = float(st.text_input("Surface bâtie (en m²)", "0.0"))
    n_rooms = float(st.text_input("Nombre de chambres", "0.0"))
    has_parking = float(st.text_input("Parking disponible (1 pour Oui, 0 pour Non)", "0.0"))

    quartiers = {
        "San Cristóbal, Madrid": 11.529439,
        "Los Ángeles, Madrid": 11.905722,
        "San Andrés, Madrid": 11.931325,
        "Los Rosales, Madrid": 11.951438,
        "Villaverde, Madrid": 11.801475,
        "Butarque, Madrid": 12.305518,
        "Vicálvaro, Madrid": 12.195680,
        "Ambroz, Madrid": 11.850769,
        "Casco Histórico de Vicálvaro, Madrid": 12.175785,
        "El Cañaveral - Los Berrocales, Madrid": 12.579838,
        "Valdebernardo - Valderribas, Madrid": 12.575724,
        "Casco Histórico de Vallecas, Madrid": 11.908568,
        "Ensanche de Vallecas - La Gavia, Madrid": 12.374956,
        "Villa de Vallecas, Madrid": 12.109924,
        "Santa Eugenia, Madrid": 12.321534,
        "Orcasitas, Madrid": 12.178719,
        "Usera, Madrid": 12.053165,
        "San Fermín, Madrid": 12.228630,
        "Pradolongo, Madrid": 11.975490,
        "Zofío, Madrid": 12.000534,
        "Almendrales, Madrid": 12.056315,
        "Moscardó, Madrid": 12.078860,
        "12 de Octubre-Orcasur, Madrid": 12.262399,
        "Valdeacederas, Madrid": 12.420474,
        "Tetuán, Madrid": 12.976287,
        "Berruguete, Madrid": 12.314153,
        "Cuatro Caminos, Madrid": 12.973625,
        "Cuzco-Castillejos, Madrid": 13.042108,
        "Bellas Vistas, Madrid": 12.354689,
        "Ventilla-Almenara, Madrid": 12.665285,
        "Retiro, Madrid": 13.390246,
        "Adelfas, Madrid": 12.827913,
        "Ibiza, Madrid": 13.287037,
        "Pacífico, Madrid": 12.859025,
        "Niño Jesús, Madrid": 13.348883,
        "Jerónimos, Madrid": 13.573377,
        "Estrella, Madrid": 13.162152,
        "Puente de Vallecas, Madrid": 11.879897,
        "Palomeras Bajas, Madrid": 11.885731,
        "San Diego, Madrid": 11.771243,
        "Palomeras sureste, Madrid": 12.071321,
        "Entrevías, Madrid": 11.788795,
        "Numancia, Madrid": 11.930122,
        "Portazgo, Madrid": 11.857501,
        "Aravaca, Madrid": 13.471373,
        "Argüelles, Madrid": 13.316595,
        "Moncloa, Madrid": 13.753433,
        "Valdezarza, Madrid": 12.475975,
        "Ciudad Universitaria, Madrid": 13.631014,
        "Fontarrón, Madrid": 12.084872,
        "Moratalaz, Madrid": 12.368257,
        "Vinateros, Madrid": 12.315080,
        "Marroquina, Madrid": 12.480875,
        "Media Legua, Madrid": 12.520585,
        "Horcajo, Madrid": 12.806211,
        "Pavones, Madrid": 12.564051,
        "Puerta del Ángel, Madrid": 12.168963,
        "Latina, Madrid": 12.131680,
        "Los Cármenes, Madrid": 12.302307,
        "Aluche, Madrid": 12.161909,
        "Águilas, Madrid": 11.989801,
        "Lucero, Madrid": 12.117311,
        "Campamento, Madrid": 12.248064,
        "Cuatro Vientos, Madrid": 12.710900,
        "Valdemarín, Madrid": 13.711245,
        "Casa de Campo, Madrid": 12.757182,
        "El Plantío, Madrid": 13.255185,
        "Fuencarral, Madrid": 13.313930,
        "Las Tablas, Madrid": 13.101345,
        "La Paz, Madrid": 13.089372,
        "Montecarmelo, Madrid": 13.357600,
        "Peñagrande, Madrid": 13.212258,
        "Tres Olivos - Valverde, Madrid": 12.492096,
        "Pilar, Madrid": 12.328591,
        "Mirasierra, Madrid": 13.610750,
        "Arroyo del Fresno, Madrid": 13.015024,
        "Fuentelarreina, Madrid": 13.416873,
        "El Pardo, Madrid": 12.750289,
        "Sanchinarro, Madrid": 13.176532,
        "Hortaleza, Madrid": 13.671296,
        "Palomas, Madrid": 13.509400,
        "Conde Orgaz-Piovera, Madrid": 13.806078,
        "Virgen del Cortijo - Manoteras, Madrid": 12.723847,
        "Valdebebas - Valdefuentes, Madrid": 13.244563,
        "Pinar del Rey, Madrid": 12.354153,
        "Canillas, Madrid": 12.695895,
        "Apóstol Santiago, Madrid": 12.667376,
        "Chamberí, Madrid": 13.558409,
        "Trafalgar, Madrid": 13.168591,
        "Nuevos Ministerios-Ríos Rosas, Madrid": 13.225055,
        "Vallehermoso, Madrid": 13.368085,
        "Almagro, Madrid": 13.650642,
        "Gaztambide, Madrid": 13.287266,
        "Arapiles, Madrid": 13.161839,
        "Ventas, Madrid": 12.181797,
        "Pueblo Nuevo, Madrid": 12.243724,
        "Atalaya, Madrid": 13.234242,
        "Quintana, Madrid": 12.347603,
        "San Juan Bautista, Madrid": 13.216958,
        "Ciudad Lineal, Madrid": 12.863423,
        "Costillares, Madrid": 13.245019,
        "Concepción, Madrid": 12.620618,
        "Colina, Madrid": 13.094352,
        "San Pascual, Madrid": 12.828080,
        "Chamartín, Madrid": 13.702813,
        "El Viso, Madrid": 13.682510,
        "Bernabéu-Hispanoamérica, Madrid": 13.415051,
        "Prosperidad, Madrid": 12.974813,
        "Nueva España, Madrid": 13.570121,
        "Castilla, Madrid": 13.243685,
        "Ciudad Jardín, Madrid": 13.078587,
        "Lavapiés-Embajadores, Madrid": 12.674573,
        "Opañel, Madrid": 12.109398,
        "Comillas, Madrid": 12.257502,
        "Abrantes, Madrid": 11.976745,
        "San Isidro, Madrid": 12.112431,
        "Carabanchel, Madrid": 12.077915,
        "Puerta Bonita, Madrid": 11.960624,
        "Vista Alegre, Madrid": 12.025508,
        "Pau de Carabanchel, Madrid": 12.551007,
        "Buena Vista, Madrid": 12.064055,
        "Huertas-Cortes, Madrid": 13.306997,
        "Malasaña-Universidad, Madrid": 12.973159,
        "Chueca-Justicia, Madrid": 13.245680,
        "Palacio, Madrid": 13.018479,
        "Centro, Madrid": 13.385319,
        "Sol, Madrid": 13.257913,
        "Barrio de Salamanca, Madrid": 13.618751,
        "Goya, Madrid": 13.467374,
        "Guindalera, Madrid": 12.982934,
        "Lista, Madrid": 13.399454,
        "Castellana, Madrid": 13.586163,
        "Fuente del Berro, Madrid": 12.865312,
        "Recoletos, Madrid": 13.949504,
        "Imperial, Madrid": 12.840645,
        "Chopera, Madrid": 12.515869,
        "Acacias, Madrid": 12.761917,
        "Delicias, Madrid": 12.719490,
        "Palos de Moguer, Madrid": 12.964373,
        "Atocha, Madrid": 13.101231,
        "Arganzuela, Madrid": 13.132510,
        "Legazpi, Madrid": 12.574795,
    }

    selected_quartier = st.selectbox("🏙️ Choisissez un quartier", list(quartiers.keys()))

    subtitle_encoded = quartiers[selected_quartier]

    df = pd.DataFrame({
        'sq_mt_built': [sq_mt_built],
        'n_rooms': [n_rooms],
        'has_parking': [has_parking],
        'subtitle_encoded': [subtitle_encoded],
    })

    df_scaled = scaler.transform(df)  
    if st.button('🔮 Soumettre'):
        result = data.predict(df_scaled)
        predicted_price = np.exp(result) / 10
        st.success(f"💶 **Le prix prédit pour cette maison à {selected_quartier} est de :**")
        st.markdown(f"<h2 style='text-align: center; color: green;'> {predicted_price[0]:,.2f} € </h2>", unsafe_allow_html=True)
        
        st.session_state['submitted'] = True

if st.session_state['submitted']:
    st.write("---")  
    st.info("🔄 Voulez-vous effectuer une nouvelle prédiction ?")
    if st.button('🔄 Commencer une nouvelle prédiction'):
        reset_form()