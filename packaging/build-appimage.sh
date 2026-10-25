#!/usr/bin/env bash
# Builds OpenCymple-<version>-x86_64.AppImage: a relocatable CPython (from uv) with the app and its dependencies
# installed into an AppDir, packed with appimagetool. 
# Needs: uv, appimagetool (APPIMAGETOOL=/path or on PATH). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/packaging/build"
APPDIR="$BUILD/OpenCymple.AppDir"
PYVER="${PYVER:-3.12}"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
OUT="$ROOT/packaging/OpenCymple-$VERSION-x86_64.AppImage"
TOOL="${APPIMAGETOOL:-$(command -v appimagetool || true)}"

[ -f "${CYMPLE_MODELS:-}/model0.bin" ] || { echo "set CYMPLE_MODELS to the folder with model0.bin ..." >&2; exit 1; }

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr"
# --no-project and a neutral cwd: otherwise uv answers with the project's .venv, which is not relocatable
mkdir -p "$BUILD"
PYBIN="$(cd "$BUILD" && uv python find --managed-python --no-project "$PYVER")"
PYHOME="$(dirname "$(dirname "$(readlink -f "$PYBIN")")")"
[ ! -e "$PYHOME/pyvenv.cfg" ] || { echo "$PYHOME is a venv, not a standalone Python" >&2; exit 1; }
cp -a "$PYHOME" "$APPDIR/usr/python"
rm -f "$APPDIR/usr/python/lib/python3."*/EXTERNALLY-MANAGED

uv pip install --python "$APPDIR/usr/python/bin/python3" --no-cache "$ROOT[gui,xr]"

uv pip uninstall --python "$APPDIR/usr/python/bin/python3" pytest pluggy iniconfig pygments pip 2>/dev/null || true

# trim: bytecode caches, test suites, headers
find "$APPDIR/usr/python" -name __pycache__ -prune -exec rm -rf {} +
find "$APPDIR/usr/python/lib" -type d \( -name tests -o -name test \) -path '*site-packages/*' -prune -exec rm -rf {} +
rm -rf "$APPDIR/usr/python/include" "$APPDIR/usr/python/share"

mkdir -p "$APPDIR/usr/share/opencymple/model"
cp "$CYMPLE_MODELS"/model*.bin "$APPDIR/usr/share/opencymple/model/"

# trim what the app never loads: pyopenxr's Android/aarch64 loaders and API layers (only the x86_64 Linux
# loader is used), and the Qt modules beyond Core/Gui/Widgets/Network/Svg (Quick, QML, Designer, ...)
SP="$(echo "$APPDIR"/usr/python/lib/python3.*/site-packages)"
for d in "$SP/xr/library" "$SP/xr/api_layer"; do     # Python files stay: xr imports them
    find "$d" -mindepth 1 -maxdepth 1 -type d ! -name linux_x86_64 ! -name __pycache__ -exec rm -rf {} +
done
rm -f "$SP"/xr/api_layer/linux_x86_64/libXrApiLayer_{core_validation,api_dump}.so   # debugging layers
PS="$SP/PySide6"
for m in Concurrent Designer Help LabsStyleKit Qml QmlFeatures Quick QuickControls2 QuickTest QuickWidgets Sql Test UiTools Xml PrintSupport; do
    rm -f "$PS/Qt$m.abi3.so" "$PS/Qt$m.pyi"
done
rm -f "$PS/Qt/lib/"libQt6{Quick,Qml,Labs,Designer,Help,Test,Sql,UiTools,Lottie,PrintSupport,QmlCompiler,QmlCore}*.so* \
      "$PS/Qt/lib/"libpyside6qml*.so* "$PS/"libpyside6qml*.so*
rm -rf "$PS/Qt/qml" "$PS/Qt/translations" "$PS/Qt/plugins"/{designer,qmltooling,sqldrivers,printsupport} \
       "$PS/include" "$PS/typesystems" "$PS/glue" "$PS/scripts" "$PS/Qt/libexec"
rm -f "$PS"/*.pyi

cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
export PYTHONNOUSERSITE=1
exec "$HERE/usr/python/bin/python3" -m opencymple.opencymple "$@"
RUN
chmod +x "$APPDIR/AppRun"

cp "$ROOT/opencymple/resources/opencymple.svg" "$APPDIR/opencymple.svg"
ln -sf opencymple.svg "$APPDIR/.DirIcon"
cat > "$APPDIR/opencymple.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=OpenCymple
GenericName=Eye and face tracking
Comment=OpenCymple eye and face tracking for VRChat
Exec=opencymple
Icon=opencymple
Terminal=false
Categories=Utility;X-WiVRn-VR;
DESK

[ -n "$TOOL" ] || { echo "AppDir ready: $APPDIR (appimagetool not found; set APPIMAGETOOL)"; exit 0; }
cd "$BUILD"
ARCH=x86_64 "$TOOL" --no-appstream "$APPDIR" "$OUT"
echo "built $OUT"
