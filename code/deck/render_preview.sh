#!/bin/bash
# render_preview.sh deck.pptx [outdir]: PDF preview with the real macOS fonts (Georgia, Avenir Next).
# Homebrew LibreOffice reads fonts through fontconfig, which does not see /System/Library/Fonts by default; without this
# it substitutes Linux Libertine, which is narrower than Georgia and hides line wraps that PowerPoint will show.
set -euo pipefail
out="${2:-$(dirname "$1")}"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT
cat > "$tmp/fonts.conf" <<CONF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/System/Library/Fonts</dir>
  <dir>/System/Library/Fonts/Supplemental</dir>
  <dir>/Library/Fonts</dir>
  <dir>$HOME/Library/Fonts</dir>
  <cachedir>$tmp/cache</cachedir>
</fontconfig>
CONF
FONTCONFIG_FILE="$tmp/fonts.conf" soffice -env:UserInstallation="file://$tmp/profile" --headless --convert-to pdf --outdir "$out" "$1" >/dev/null 2>&1
echo "wrote $out/$(basename "${1%.pptx}").pdf"
