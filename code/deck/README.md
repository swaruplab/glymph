# Deck sources

Builds `reports/Glymph_overview_dark.pptx` from the benchmark tables.

```bash
python3 code/deck/make_deck_figures.py dark
python3 code/deck/build_glymph_deck.py dark
python3 code/deck/check_package.py reports/Glymph_overview_dark.pptx
```

Fonts: Georgia and Avenir Next (macOS). Icons in `icons/` are Phosphor Icons (https://phosphoricons.com), MIT License, recoloured.
