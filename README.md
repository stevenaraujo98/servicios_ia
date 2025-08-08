# Servicion de Inteligencia Artificial

FastApi

- enlace: https://fastapi.tiangolo.com/deployment/docker/#check-it

### Docker
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


### Dev docker compose
```
sudo docker compose -f docker-compose.dev.yml up -d --build
```

### Test docker compose
```
sudo docker compose -f docker-compose.test.yml up -d --build
```

### Prod docker compose
```
sudo docker compose -f docker-compose.prod.yml up -d --build
```

### Down the containers compose kill all
```
<!-- Por si no funciona solo down  -->
sudo docker compose -f docker-compose.dev.yml down 

<!-- Elimina tambien la imagen creada  -->
sudo docker compose -f docker-compose.dev.yml down --rmi all 

<!-- El down solo -->
sudo docker compose down
```


### Para monitoreo
```
sudo docker logs containerai-test
sudo docker logs containerai-prod

<!-- seguimiento en tiempo real -->
sudo docker logs -f containerai-test  
sudo docker logs -f containerai-prod  

sudo docker ps -a
```


### Para correr nuevamente
```
<!-- Detener el contenedor o el kill -->
sudo docker stop containerai

<!-- Eliminar contenedor -->
sudo docker rm containerai

<!-- Eliminar imagen -->
sudo docker rmi serviceai 

sudo docker build -t serviceai .
sudo docker run -d --name containerai --restart unless-stopped -p 80:80 serviceai
```

## Host: http://localhost:8080

#### Consideraciones en linux
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
```


#### APIS
- /predict/distilbert_10e_24b_0


#### Alternativa de deploy
- requirement
    uvicorn[standard]
- Dockerfile
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"] 
    CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"] 
