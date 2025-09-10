limit_min = 50

# --- Configuración RABBIT MQ ---
RABBITMQ_HOST="rabbitmq"
RABBITMQ_USERNAME="guest"
RABBITMQ_PASSWORD="guest"
RABBITMQ_PORT="5672"

# El broker apunta al servicio "rabbitmq" de tu docker-compose.
broker_url = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}//"

# --- Configuración REDIS ---
REDIS_HOST="redis"
REDIS_PORT=6379
REDIS_CELERY_DB_INDEX=10
REDIS_STORE_DB_INDEX=0

# El backend de resultados usa RPC, que también funciona sobre RabbitMQ.
# Para producción a gran escala, se suele preferir Redis.
# result_backend = "rpc://"
result_backend = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_CELERY_DB_INDEX}"

# Se establece el tiempo en segundos. 7 días = 7 * 24 * 60 * 60 = 604800 segundos.
result_expires = 604800

stages = ["Processing", "confirmed", "shipped", "in transit", "arrived", "delivered"]
