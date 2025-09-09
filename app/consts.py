limit_min = 50
# --- Configuración RABBIT MQ ---
RABBITMQ_HOST="rabbitmq"
RABBITMQ_USERNAME="guest"
RABBITMQ_PASSWORD="guest"
RABBITMQ_PORT="5672"


# El broker apunta al servicio "rabbitmq" de tu docker-compose.
BROKER_URL = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}//"
# El backend de resultados usa RPC, que también funciona sobre RabbitMQ.
# Para producción a gran escala, se suele preferir Redis.
RESULT_BACKEND = "rpc://"

# REDIS_HOST=kredis
REDIS_PORT=6379
REDIS_CELERY_DB_INDEX=10
REDIS_STORE_DB_INDEX=0

stages = ["Processing", "confirmed", "shipped", "in transit", "arrived", "delivered"]
