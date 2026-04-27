#!/usr/bin/env bash
set -euo pipefail

echo "== System =="
uname -a || true
echo "arch: $(uname -m)"

echo
echo "== Jetson / L4T packages =="
dpkg-query --showformat='nvidia-l4t-core: ${Version}\n' --show nvidia-l4t-core 2>/dev/null || echo "nvidia-l4t-core: not found"
dpkg-query --showformat='nvidia-jetpack: ${Version}\n' --show nvidia-jetpack 2>/dev/null || echo "nvidia-jetpack: not found"

echo
echo "== Python / uv =="
python3 --version || true
uv --version || true

echo
echo "== uv venv python =="
if [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python - <<'PY'
import platform
print('python:', platform.python_version())
print('machine:', platform.machine())
PY
else
  echo ".venv/bin/python not found"
fi

echo
echo "== NVIDIA PyTorch index probes =="
for jp in v62 v61 v60 v51; do
  url="https://developer.download.nvidia.com/compute/redist/jp/${jp}/pytorch/"
  code=$(curl -L -s -o /dev/null -w "%{http_code}" "$url" || true)
  echo "$jp: $code $url"
  if [[ "$code" == "200" ]]; then
    echo "  cp310 wheels (top 5):"
    curl -fsSL "$url" | grep -o 'torch-[^"<]*cp310[^"<]*\.whl' | head -n 5 || true
  fi
done

echo
echo "Done. Use the newest reachable jp/vXX path that has cp310 wheel files."
