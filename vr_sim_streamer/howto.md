
run the app
```
MUJOCO_GL=egl uvicorn server.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile=certs/key.pem --ssl-certfile=certs/cert.pem
```

Create certificate
```
mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem -days 365 -subj "/CN=192.168.178.32" -addext "subjectAltName=IP:192.168.178.32,DNS:localhost"  
```