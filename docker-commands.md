# Docker Commands

Brief reference for the Docker CLI commands used by this project. Examples use PowerShell syntax.

## Image Names

- Default Web UI, without Blender: `anthonykatona/optical-profilometer:latest`
- Versioned default Web UI: `anthonykatona/optical-profilometer:1.2`
- Blender-enabled Web UI: `anthonykatona/optical-profilometer:latest-blender`
- Versioned Blender-enabled Web UI: `anthonykatona/optical-profilometer:1.2-blender`

## Build Local Images

Build the default Web UI image:

```powershell
docker build -t optical-profilometer .
```

Build the Web UI with Blender installed:

```powershell
docker build --target blender -t optical-profilometer-blender .
```

Build the command-line analysis image:

```powershell
docker build --target runtime -t optical-profilometer-cli .
```

Build the standalone Blender OBJ renderer:

```powershell
docker build --target blender-renderer -t optical-profilometer-renderer .
```

## Run the Web UI

Run a local default Web UI container on `http://localhost:8000`:

```powershell
New-Item -ItemType Directory -Force .\webui-data | Out-Null

docker run --rm -p 8000:8000 `
  --mount "type=bind,source=${PWD}\webui-data,target=/app/webui_data" `
  optical-profilometer
```

Run the DockerHub `latest` image instead of a local build:

```powershell
docker run --rm -p 8000:8000 `
  --mount "type=bind,source=${PWD}\webui-data,target=/app/webui_data" `
  anthonykatona/optical-profilometer:latest
```

Run the Blender-enabled Web UI:

```powershell
docker run --rm -p 8000:8000 `
  --mount "type=bind,source=${PWD}\webui-data,target=/app/webui_data" `
  anthonykatona/optical-profilometer:latest-blender
```

## Run CLI Analysis

Run analysis against a mounted XYZ file and write outputs to a mounted output folder:

```powershell
New-Item -ItemType Directory -Force .\docker-out | Out-Null

docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-cli /data/test.xyz -o /out --no-display
```

Export OBJ files during CLI analysis:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-cli /data/test.xyz -o /out --no-display --export-obj roughness
```

## Run Standalone Blender Rendering

Render an existing OBJ with the standalone renderer image:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-renderer `
  --input /out/test_roughness.obj `
  --output /out/test_roughness_render.png
```

## Multi-Architecture Build and Push

Check available builders and supported platforms:

```powershell
docker buildx ls
```

Build and push both Web UI images for `linux/amd64` and `linux/arm64` using `docker-bake.hcl`:

```powershell
$env:REGISTRY_IMAGE="anthonykatona/optical-profilometer"
$env:VERSION="1.2"
docker buildx bake --push
```

This publishes:

- `anthonykatona/optical-profilometer:1.2`
- `anthonykatona/optical-profilometer:latest`
- `anthonykatona/optical-profilometer:1.2-blender`
- `anthonykatona/optical-profilometer:latest-blender`

## Verify DockerHub Manifests

Inspect the remote multi-architecture manifests:

```powershell
docker buildx imagetools inspect anthonykatona/optical-profilometer:latest
docker buildx imagetools inspect anthonykatona/optical-profilometer:latest-blender
docker buildx imagetools inspect anthonykatona/optical-profilometer:1.2
docker buildx imagetools inspect anthonykatona/optical-profilometer:1.2-blender
```

Look for `linux/amd64` and `linux/arm64` under `Manifests`.

## Container Utilities

Run the Web UI in the background:

```powershell
docker run -d --name optical-profilometer-web -p 8000:8000 `
  --mount "type=bind,source=${PWD}\webui-data,target=/app/webui_data" `
  anthonykatona/optical-profilometer:latest
```

View running containers:

```powershell
docker ps
```

View logs:

```powershell
docker logs optical-profilometer-web
```

Stop a background container:

```powershell
docker stop optical-profilometer-web
```

Pull the latest published images:

```powershell
docker pull anthonykatona/optical-profilometer:latest
docker pull anthonykatona/optical-profilometer:latest-blender
```
