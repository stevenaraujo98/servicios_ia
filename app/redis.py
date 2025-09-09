from fastapi import WebSocket
from typing import Dict

# =================================================================
# --- SECCIÓN DE WEBSOCKETS Y NOTIFICACIONES EN TIEMPO REAL ---
# =================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"[DIAGNÓSTICO] Cliente conectado: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        print(f"[DIAGNÓSTICO] Cliente desconectado: {client_id}")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
        else:
            print(f"[DIAGNÓSTICO WARN] Intento de enviar mensaje a cliente desconectado: {client_id}")
