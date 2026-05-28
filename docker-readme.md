# Docker Guide For This Project

This guide is for someone who is new to Docker and wants to run this optical profilometry pipeline safely and predictably.

It covers:

- what the Docker images do
- how to build and run them locally
- where your input and output files live
- how to run many containers at once without stepping on your own files
- how to use the optional headless Blender renderer

All examples below use Windows PowerShell, because this repository is currently being used from Windows.

## 1. What You Need

Before anything else:

- Install Docker Desktop.
- Start Docker Desktop and wait until it says Docker is running.
- Open PowerShell in the repository root.

You should be in the folder that contains:

- `Dockerfile`
- `analyze_heightmap.py`
- `test.xyz`

If you want to verify Docker is available:

```powershell
docker --version
docker info
```

If `docker info` fails, Docker Desktop is usually not running yet.

## 2. Very Short Docker Explanation

You only need three ideas for this project:

- An `image` is the packaged environment.
- A `container` is one temporary run of that image.
- A `bind mount` connects a real folder or file on your computer into the container.

This project uses bind mounts so that:

- your `.xyz` input files stay on your computer
- the container reads them from `/data`
- the container writes results into `/out`

## 3. The Two Images In This Repository

This repository can build two different images from the same `Dockerfile`.

### Main analysis image

Image name used in this guide:

```powershell
optical-profilometer
```

What it does:

- reads `.xyz` data
- computes stats
- saves plots
- exports OBJ files for Blender

What it does not do:

- open interactive windows
- render Blender scenes

### Optional Blender renderer image

Image name used in this guide:

```powershell
optical-profilometer-blender
```

What it does:

- imports an OBJ file
- builds a basic Blender scene
- renders a still image

What it does not do:

- run the profilometry analysis itself

## 4. Important Paths: Where Files Actually Are

The most important beginner question is: where do the files end up?

This project uses these container paths:

| Container path | Meaning |
| --- | --- |
| `/app` | The code inside the image |
| `/data` | Your input file or input folder from the host |
| `/out` | Your output folder from the host |

Example host-to-container mapping:

| On your computer | Inside the container |
| --- | --- |
| `${PWD}\test.xyz` | `/data/test.xyz` |
| `${PWD}\heightmaps` | `/data` |
| `${PWD}\docker-out` | `/out` |

### What gets written where

If you run the analysis container with:

```powershell
--mount "type=bind,source=${PWD}\docker-out,target=/out"
```

then files written to `/out` inside the container will appear in:

```powershell
.\docker-out
```

on your computer.

Typical output files are:

- `test_statistics.txt`
- `test_analysis.png`
- `test_roughness.obj`
- `test_form.obj`
- `test_raw.obj`

If you use the Blender renderer, the rendered image will also appear in your mounted host output folder, for example:

- `test_roughness_render.png`

### Very important

If you do **not** mount an output folder, files written inside the container are temporary and will disappear when the container is removed.

That is why this project should normally be run with an `/out` bind mount.

## 5. Build The Images

### Build the main analysis image

Run this from the repository root:

```powershell
docker build -t optical-profilometer .
```

This builds the default final stage from the `Dockerfile`.

### Build the optional Blender image

Run this only if you want the rendering step:

```powershell
docker build --target blender-renderer -t optical-profilometer-blender .
```

This image is larger because it installs Blender.

## 6. First Run: Smallest Possible Test

Create a host output folder first:

```powershell
New-Item -ItemType Directory -Force .\docker-out | Out-Null
```

### Show the main container help

```powershell
docker run --rm optical-profilometer
```

What `--rm` means:

- remove the container automatically after it exits

### Run a fast stats-only smoke test

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  optical-profilometer /data/test.xyz --stats-only -r 32
```

What this does:

- mounts `test.xyz` into the container
- runs the analysis script
- uses `-r 32` to make the run much faster
- prints stats to the terminal

This command does **not** save files, because no `/out` folder was mounted.

## 7. First Real Run: Save Files To Your Computer

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer /data/test.xyz -o /out --no-display -r 32
```

This is the standard container pattern for this project.

What it does:

- reads `/data/test.xyz`
- saves outputs into `/out`
- writes those files back into `.\docker-out` on your computer
- avoids any interactive display

After it finishes, check:

```powershell
Get-ChildItem .\docker-out
```

You should see files like:

- `test_statistics.txt`
- `test_analysis.png`

## 8. Why `--no-display` And `-o /out` Go Together

Inside the container, this project is headless. That means:

- it cannot open a Matplotlib window for you
- it should save plots to disk instead

So the correct pattern is:

```powershell
-o /out --no-display
```

This repository now enforces that. If you pass `--no-display` without `-o`, the command will fail on purpose.

## 9. Running On A Real Dataset Folder

Instead of mounting one file, you can mount a whole folder of `.xyz` files.

Example:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\heightmaps,target=/data,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer `
  /data/2026-03_PCD-1um-pristine/low_zoom_01.xyz `
  -o /out --no-display
```

The rule is simple:

- left side of `source=` is the real host path
- right side of `target=` is the in-container path you will pass to the script

## 10. Export OBJ Files For Blender

If you want geometry for rendering or further 3D work:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer `
  /data/test.xyz -o /out --no-display --export-obj roughness -r 32
```

After it finishes, you should see:

- `test_statistics.txt`
- `test_analysis.png`
- `test_roughness.obj`

If you export more than one map:

```powershell
--export-obj raw form roughness waviness+roughness
```

then you will get one OBJ file per exported surface.

## 11. Render An OBJ With The Blender Container

First build the Blender image if you have not already:

```powershell
docker build --target blender-renderer -t optical-profilometer-blender .
```

Then render an OBJ:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-blender `
  --input /out/test_roughness.obj `
  --output /out/test_roughness_render.png
```

This reads:

- `.\docker-out\test_roughness.obj`

and writes:

- `.\docker-out\test_roughness_render.png`

### Useful optional Blender flags

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-blender `
  --input /out/test_roughness.obj `
  --output /out/test_roughness_render.png `
  --engine CYCLES `
  --resolution-x 1920 `
  --resolution-y 1080 `
  --samples 128 `
  --camera-azimuth 45 `
  --camera-elevation 30 `
  --transparent-background
```

## 12. Local Workflow: Recommended Folder Layout

For repeated runs, a simple host layout helps:

```text
Optical_Profilometer_XYZ_Workup_Pipeline/
  heightmaps/
  docker-out/
    run-01/
    run-02/
    render-01/
```

Why this helps:

- each run stays separate
- files are easier to compare later
- parallel containers do not overwrite each other

## 13. Running Multiple Containers In Parallel

Yes, you can run multiple analysis containers at once.

This is safe **if**:

- each container reads different inputs, or you understand they are reading the same file
- each container writes to a different host output folder

### Do not do this

Do not point several runs at the same output folder if they will create the same filenames.

For example, two containers both processing `test.xyz` into the same `docker-out` folder will both try to write:

- `test_statistics.txt`
- `test_analysis.png`

One run may overwrite the other.

### Safe parallel pattern

Create one output folder per run:

```powershell
New-Item -ItemType Directory -Force .\docker-out\run-a | Out-Null
New-Item -ItemType Directory -Force .\docker-out\run-b | Out-Null
```

Then start two detached containers:

```powershell
docker run -d --name profilometer-a `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out\run-a,target=/out" `
  optical-profilometer `
  /data/test.xyz -o /out --no-display -r 16

docker run -d --name profilometer-b `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out\run-b,target=/out" `
  optical-profilometer `
  /data/test.xyz -o /out --no-display -r 32
```

What `-d` means:

- run in the background

### Check running containers

```powershell
docker ps
```

### Watch logs

```powershell
docker logs -f profilometer-a
docker logs -f profilometer-b
```

### Inspect outputs

```powershell
Get-ChildItem .\docker-out\run-a
Get-ChildItem .\docker-out\run-b
```

### Remove finished named containers

```powershell
docker rm profilometer-a profilometer-b
```

If you prefer automatic cleanup, keep using `--rm` and launch runs from separate terminals instead of `-d`.

## 14. Parallel Blender Renders

You can also run multiple Blender render containers at once.

Same rule:

- one output folder or unique output filenames per render job

Example:

```powershell
New-Item -ItemType Directory -Force .\docker-out\render-a | Out-Null
New-Item -ItemType Directory -Force .\docker-out\render-b | Out-Null

Copy-Item .\docker-out\test_roughness.obj .\docker-out\render-a\test_roughness.obj
Copy-Item .\docker-out\test_roughness.obj .\docker-out\render-b\test_roughness.obj

docker run -d --name blender-a `
  --mount "type=bind,source=${PWD}\docker-out\render-a,target=/out" `
  optical-profilometer-blender `
  --input /out/test_roughness.obj `
  --output /out/render-a.png

docker run -d --name blender-b `
  --mount "type=bind,source=${PWD}\docker-out\render-b,target=/out" `
  optical-profilometer-blender `
  --input /out/test_roughness.obj `
  --output /out/render-b.png `
  --camera-azimuth 55
```

## 15. Performance Notes

Running more containers at once does not automatically make the machine faster.

Usually:

- more parallel analysis containers means more CPU and RAM usage
- more parallel Blender containers means much more CPU and RAM usage
- very dense OBJ files can be expensive to render

Practical advice:

- start with one container
- then try two in parallel
- check whether total runtime actually improves

On Windows with Docker Desktop, file access is often faster if your working data is in the WSL/Linux filesystem instead of a OneDrive-backed folder.

## 16. Common Problems

### Problem: `docker info` fails

Cause:

- Docker Desktop is not running

Fix:

- start Docker Desktop
- wait until Docker is ready

### Problem: the container says the input file does not exist

Cause:

- the host file path in `source=` is wrong
- the container path you passed to the script does not match the mount target

Fix:

- check both the `source=` path and the `/data/...` path

### Problem: no files show up on the host

Cause:

- you forgot the `/out` bind mount
- or you forgot `-o /out`

Fix:

- mount a host output folder to `/out`
- pass `-o /out --no-display`

### Problem: a run overwrote an older run

Cause:

- both runs wrote into the same host output folder using the same filenames

Fix:

- use one output folder per run

### Problem: Blender image build is slow

Cause:

- Blender is a large dependency

Fix:

- this is normal
- only build the Blender image if you need it

## 17. Useful Docker Commands

Show built images:

```powershell
docker image ls
```

Show running containers:

```powershell
docker ps
```

Show all containers, including stopped ones:

```powershell
docker ps -a
```

Follow logs from a background container:

```powershell
docker logs -f <container-name>
```

Remove one or more stopped containers:

```powershell
docker rm <container-name>
```

Remove unused stopped containers:

```powershell
docker container prune
```

## 18. Recommended Beginner Workflow

If you only want the safest basic path, do this:

1. Start Docker Desktop.
2. Run `docker build -t optical-profilometer .`
3. Create `.\docker-out`
4. Run one stats-only test on `test.xyz`
5. Run one saved-output analysis with `-o /out --no-display`
6. Confirm the files exist in `.\docker-out`
7. Only after that, try OBJ export and Blender rendering
8. Only after that, try parallel jobs with separate output folders

## 19. Copy-Paste Cheat Sheet

Build the main image:

```powershell
docker build -t optical-profilometer .
```

Build the Blender image:

```powershell
docker build --target blender-renderer -t optical-profilometer-blender .
```

Create output folder:

```powershell
New-Item -ItemType Directory -Force .\docker-out | Out-Null
```

Fast test:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  optical-profilometer /data/test.xyz --stats-only -r 32
```

Save analysis outputs:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer /data/test.xyz -o /out --no-display -r 32
```

Export OBJ:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\test.xyz,target=/data/test.xyz,readonly" `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer /data/test.xyz -o /out --no-display --export-obj roughness -r 32
```

Render OBJ:

```powershell
docker run --rm `
  --mount "type=bind,source=${PWD}\docker-out,target=/out" `
  optical-profilometer-blender `
  --input /out/test_roughness.obj `
  --output /out/test_roughness_render.png
```
