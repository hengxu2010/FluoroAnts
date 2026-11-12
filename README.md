# FluoroAnts

FluoroAnts is a Python desktop application for detecting and quantifying positive regions in 2-D fluorescence images. It provides English and Chinese interfaces, configurable detection settings, and batch processing. Image processing runs locally.

## Install and run

Requires Python 3.10 or later, NumPy, OpenCV, SciPy, and PySide6. From the repository directory, run:

```sh
python -m pip install -r requirements.txt
python app.py
```

Use `python app.py --lang en` or `python app.py --lang zh` to select the interface language.

## Single-file launcher

`FluoroAnts_v6.3.pyw` embeds the same application source as this repository. After installing the dependencies above, run the command below, or double-click it on Windows when Python file associations are configured. This is a Python script and requires Python and the listed packages.

```sh
python FluoroAnts_v6.3.pyw
```

## Batch processing

```sh
python app.py "input_images" --batch --output "output_results"
```

Replace the example paths and keep the output directory outside the input directory. Add `--recursive` to include subdirectories. Use `python app.py --help` for available options and `python app.py --list-presets` for built-in presets.

## Versions and files

- Application release: **2.12.0** (`fluoro_ants/version.py`).
- Recognition core: **2.7.1**.
- Bundled model: **v6.3**. The supplied launcher filename also uses `v6.3`; it is separate from the application release number.
- `app.py`: Application entry point.
- `fluoro_ants/`: Application modules, interface assets, and model parameters.
- `FluoroAnts_v6.3.pyw`: Single-file launcher.
