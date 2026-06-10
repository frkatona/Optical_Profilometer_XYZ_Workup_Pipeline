variable "REGISTRY_IMAGE" {
  default = "anthonykatona/optical-profilometer"
}

variable "VERSION" {
  default = "1.1"
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
