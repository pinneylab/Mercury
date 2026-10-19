#!/usr/bin/env bash
# Script to render Mermaid diagram sources (.mmd) into PNGs
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "Rendering analysis_pipeline.png..."
npx -y @mermaid-js/mermaid-cli -p puppeteer-config.json -i analysis_pipeline.mmd -o analysis_pipeline.png -b white -s 2

echo "Rendering processing_pipeline.png..."
npx -y @mermaid-js/mermaid-cli -p puppeteer-config.json -i processing_pipeline.mmd -o processing_pipeline.png -b white -s 2

echo "Done! Rendered images updated in $DIR"
