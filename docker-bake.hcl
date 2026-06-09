variable "REGISTRY_IMAGE" {
  default = "optical-profilometer"
}

variable "VERSION" {
  default = "1.0"
}

group "default" {
  targets = ["web-ui", "blender"]
}

target "web-ui" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["${REGISTRY_IMAGE}:${VERSION}"]
}

target "blender" {
  context = "."
  dockerfile = "Dockerfile"
  target = "blender"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["${REGISTRY_IMAGE}:${VERSION}-blender"]
}
