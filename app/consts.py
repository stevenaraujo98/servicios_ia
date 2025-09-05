limit_min = 50
# --- Configuración Clave ---
# El broker apunta al servicio 'rabbitmq' de tu docker-compose.
BROKER_URL = 'amqp://guest:guest@rabbitmq:5672//'
# El backend de resultados usa RPC, que también funciona sobre RabbitMQ.
# Para producción a gran escala, se suele preferir Redis.
RESULT_BACKEND = 'rpc://'