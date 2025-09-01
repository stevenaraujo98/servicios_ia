# Servicion de Inteligencia Artificial

FastApi
- enlace: https://fastapi.tiangolo.com/deployment/docker/#check-it

## Estructura
- certs
- app
    - models
        - ModelLoader.py: clase para carga de modelos tradicional y transformer. Y cargar demas modelos como detección de idioma, traducción, y nlp
        - carrera: modelos
        - ods: modelos
        - patente: modelos
    - main.py: métodos apis
    - consts.py: constantes
    - modelsEntity.py: tipos de datos peticion y respuesta
    - ods.py: metodo para uso del modelo
    - patentes.py: metodos para uso del modelo
    - carrera.py: metodos para uso del modelo
    - validations.py: validaciones utiles

## Docker
- Local
```
docker build -t serviceai .

docker run -d --name containerai -p 8080:80 serviceai
docker run -d --rm -it --name containerai -p 8080:80 serviceai
docker run -v .:/app -d --name containerai -p 8080:80 serviceai
```

### Publicar
```
sudo docker build -t serviceai .

sudo docker run -d --name containerai --restart unless-stopped -p 8080:80 serviceai

sudo docker ps
curl http://localhost
```

### Build and start (para desarrollo o test es mejor)
```
docker compose -f docker-compose.dev.yml up -d --build
sudo docker compose -f docker-compose.test.yml up -d --build
```

### Si se quiere asegurar el build por el compose
```
sudo docker-compose -f docker-compose.prod.yml up -d --force-recreate
```

#### Para produccion mejor
```
sudo docker compose -f docker-compose.prod.yml build
sudo docker compose -f docker-compose.prod.yml up -d
```

### Down the containers compose kill all
```
<!-- Por si no funciona solo down  -->
docker compose -f docker-compose.dev.yml down 
sudo docker compose -f docker-compose.test.yml down 
sudo docker compose -f docker-compose.prod.yml down 

<!-- Elimina tambien la imagen creada  -->
docker compose -f docker-compose.dev.yml down --rmi all 
sudo docker compose -f docker-compose.test.yml down --rmi all 
sudo docker compose -f docker-compose.prod.yml down --rmi all 

<!-- Elimina todo contenedor, imagen y volumen  -->
docker compose -f docker-compose.dev.yml down --rmi all -v
sudo docker compose -f docker-compose.test.yml down --rmi all -v
sudo docker compose -f docker-compose.prod.yml down --rmi all -v

<!-- El down solo -->
docker compose down
<!-- Elimina contenedor y volumen  -->
docker compose -f docker-compose.test.yml down -v
<!-- Borrar solo volumen -->
docker volume rm servicios_ia_ollama_data
```

### Restart reiniar containers detiene e inicia sin eliminar 
```
docker compose -f docker-compose.dev.yml restart
sudo docker compose -f docker-compose.test.yml restart
sudo docker compose -f docker-compose.prod.yml restart
```

### Stop containers sin eliminar los contenedores
```
docker compose -f docker-compose.dev.yml stop
sudo docker compose -f docker-compose.test.yml stop
sudo docker compose -f docker-compose.prod.yml stop
```

### Start again without build
```
docker compose -f docker-compose.dev.yml up -d
sudo docker compose -f docker-compose.test.yml up -d
sudo docker compose -f docker-compose.prod.yml up -d
```

### Para monitoreo de contenedores
```
sudo docker logs containerai-test
sudo docker logs containerai-prod

<!-- seguimiento en tiempo real -->
sudo docker logs -f containerai-test
sudo docker logs -f containerai-prod
```

### Para gestion de todo Docker
```
<!-- Ver contenedores activos -->
sudo docker ps -a
<!-- Ver las imagenes -->
sudo docker image ls
sudo docker image rm ID
<!-- Ver los volumenes -->
sudo docker volume ls
sudo docker volume rm ID

<!-- Ver consola del contenedor ejecutandose -->
<!-- Para salir de la consola del contenedor y volver a la terminal de tu host, escribe exit o presiona Ctrl + D -->
sudo docker exec -it ollama-test /bin/bash
```

### Para correr nuevamente
```
<!-- Detener el contenedor o el kill -->
sudo docker stop containerai

<!-- Reiniciar el contenedor
sudo docker restart nginx-prod

<!-- Eliminar contenedor -->
sudo docker rm containerai

<!-- Eliminar imagen -->
sudo docker rmi serviceai 

sudo docker build -t serviceai .
sudo docker run -d --name containerai --restart unless-stopped -p 80:80 serviceai
```

## Hosts:
- Dev: http://localhost:8000
- Test: http://192.168.10.37/
- Prod: modelosia.espol.edu.ec 
        200.10.147.97

## Consideraciones en linux
```
# 1. Conectarte al servidor
ssh mania@192.168.10.37

# 2. Instalar Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker mania

# 3. Subir tu código (usando scp desde tu máquina local)
scp -r tu-proyecto/ mania@192.168.10.37:~/

# 4. En el servidor, construir la imagen
cd ~/tu-proyecto
sudo docker build -t serviceai .

# 5. Ejecutar el contenedor
sudo docker run -d --name containerai --restart unless-stopped -p 80:80 serviceai

# 6. Verificar que esté funcionando
sudo docker ps
curl http://localhost
curl https://modelosia.espol.edu.ec
curl -v https://modelosia.espol.edu.ec
```

## APIS
- /docs
- /predict/ods/
- /predict/patente/
- /predict/carrera/
- /predict/objetivo/

## Alternativa de deploy
- requirement
    uvicorn[standard]
- Dockerfile
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"] 
    CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"] 

### Forma usada en Dockerfile
    CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--host", "0.0.0.0"]


## Ollama
Descargar modelos
- desde solo consola
```
docker exec ollama-dev ollama pull nombredelmodelo
```

- desde Dockerfile
```
FROM ollama/ollama

RUN ollama serve & \
    sleep 10 && \
    # 8b
    ollama pull qwen3 && \
    # 12 b
    ollama pull gemma3:12b && \
    # 4b
    ollama pull gemma3
```

## Consideracion adicional 
##### DNS
8.8.8.8 y 8.8.4.4 son los servidores DNS públicos de Google. Puedes usar otros como 1.1.1.1 (Cloudflare).
```
sudo nano /etc/docker/daemon.json

{
  "dns": ["192.168.1.17", "192.168.1.19"]
}

sudo systemctl restart docker
```

##### Certificados
```
cat dominio.crt validacion_CA.crt certification_Authority.crt > espol_bundle.crt

cat espol_bundle.crt
```

```
-----END CERTIFICATE-----   <-- Hay un salto de línea aquí
-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----   <-- Y aquí también
-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----
```
