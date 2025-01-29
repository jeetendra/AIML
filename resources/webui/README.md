# Ollama Web UI Setup Instructions

## Prerequisites
- Docker Desktop installed and running
- Ollama installed on your host system
- NVIDIA GPU with updated drivers (if using GPU acceleration)

## Setup Steps

1. **Start Ollama on your system**
   ```bash
   ollama serve
   ```

2. **Verify Docker Desktop**
   - Ensure Docker Desktop is running
   - Check WSL integration is enabled in Docker Desktop settings
   - Make sure virtualization is enabled in BIOS

3. **Start the containers**
   ```bash
   docker-compose up -d
   ```

## Accessing the Web UIs

- Ollama Web UI: http://localhost:3001
- CUDA Web UI: http://localhost:3002

## Available Models

To download and use models:
1. Open terminal and run:
   ```bash
   ollama pull <model_name>
   ```
2. Example models:
   - llama2
   - mistral
   - codellama
   - neural-chat

## Troubleshooting

1. **Connection Issues**
   - Ensure Ollama is running on port 11434
   - Check Docker Desktop status
   - Run `docker ps` to verify containers are running

2. **GPU Issues**
   - Verify NVIDIA drivers are up to date
   - Check `nvidia-smi` output
   - Ensure GPU is recognized in Docker Desktop

3. **Container Logs**
   ```bash
   docker logs open-webui-ollama
   docker logs open-webui-cuda
   ```

## Stopping the Services
```bash
docker-compose down
```

## Data Persistence
- UI settings and chat history are stored in the `open-webui` volume
- Models downloaded through Ollama are stored on your host system
