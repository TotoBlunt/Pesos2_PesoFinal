from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime

st.title("Predicción del peso final de pollos con modelos de ensamble")

# Subir archivo
upload_file = st.file_uploader('Sube un archivo Excel', type=['xlsx'])

if upload_file is not None:
    try:
        df = pd.read_excel(upload_file)
        st.write('### Vista previa de los datos')
        st.write(df.head())

        # División de datos
        X = df[['PesoSem1', 'PesoSem2', 'PesoSem3', 'PesoSem4']]
        y = df['PesoFinal']

        # Generación de características polinómicas
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Escalado de datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        # Selección de características
        selector = SelectKBest(score_func=mutual_info_regression, k=8)
        X_selected = selector.fit_transform(X_scaled, y)

        # División de los datos
        x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

        # Búsqueda de hiperparámetros para RandomForest
        rf = RandomForestRegressor(random_state=42)
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10]
        }
        grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
        grid_rf.fit(x_train, y_train)

        # Búsqueda de hiperparámetros para GradientBoosting
        gb = GradientBoostingRegressor(random_state=42)
        param_grid_gb = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='r2')
        grid_gb.fit(x_train, y_train)

        # Modelos con mejores hiperparámetros
        models = [
            ("decision_tree", DecisionTreeRegressor(max_depth=6, random_state=42)),
            ("linear_regression", LinearRegression()),
            ("k_neighbors", KNeighborsRegressor(n_neighbors=7, weights='distance')),
            ("random_forest", grid_rf.best_estimator_),
            ("gradient_boosting", grid_gb.best_estimator_)
        ]

        # Ensamble
        ensemble_model = VotingRegressor(models)
        ensemble_model.fit(x_train, y_train)

        y_pred = ensemble_model.predict(x_test)

        # Página seleccionada
        page = st.selectbox("### Selecciona una opción", ["Predicción", "Gráfico de comparación", "Métricas"])

        if page == "Métricas":
            st.write("### Métricas del modelo:")
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"- Coeficiente de determinación (R²): {r2:.4f}")
            st.write(f"- Error cuadrático medio (MSE): {mse:.4f}")
            st.write(f"- Error absoluto medio (MAE): {mae:.4f}")
            cv_scores = cross_val_score(ensemble_model, x_train, y_train, cv=5, scoring='r2')
            st.write(f"- R² promedio en validación cruzada: {cv_scores.mean():.4f}")

        elif page == "Gráfico de comparación":
            st.write("### Comparación entre valores reales y predichos")
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label="Real", color="blue")
            ax.plot(y_pred, label="Predicho", color="red")
            ax.legend()
            st.pyplot(fig)

        elif page == "Predicción":
            st.write("### Predicción de Peso Final")
            peso1 = st.number_input("Peso Semana 1", format="%.2f")
            peso2 = st.number_input("Peso Semana 2", format="%.2f")
            peso3 = st.number_input("Peso Semana 3", format="%.2f")
            peso4 = st.number_input("Peso Semana 4", format="%.2f")

            if st.button("Realizar predicción"):
                input_data = np.array([[peso1, peso2, peso3, peso4]])
                input_data_poly = poly.transform(input_data)
                input_data_scaled = scaler.transform(input_data_poly)
                input_data_selected = selector.transform(input_data_scaled)
                prediction = ensemble_model.predict(input_data_selected)[0]
                st.write(f"El peso final predicho es: {prediction:.2f} kg")

    except Exception as e:
        st.error(f"Error: {e}")
