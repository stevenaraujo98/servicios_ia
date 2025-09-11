def analizar_sentimiento_texto(texto: str) -> dict:
    """
    Función pesada que simula un análisis de sentimiento.
    """
    print(f"Iniciando análisis de sentimiento para: '{texto[:30]}...'")
    # Aquí iría la lógica real (ej: llamar a un modelo)
    import time
    time.sleep(10) # Simula una tarea larga

    if "mal" in texto:
        resultado = {"sentimiento": "negativo", "confianza": 0.95}
    else:
        resultado = {"sentimiento": "positivo", "confianza": 0.98}
    
    print("Análisis de sentimiento completado.")
    return resultado
