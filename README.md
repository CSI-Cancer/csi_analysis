# csi_analysis

This package contains classes and functions for running modular analysis pipelines on
data types commonly used by CSI-Cancer. Currently, this only includes whole-slide
images (immunoflourescent scans). In particular, this package is meant to provide
abstract base classes that can be fit into an then run in a standard pipeline. This will
enable more interchangeability and easier development of new modules, such as for image
segmentation or feature extraction.

While much of the functionality is specific to the CSI-Cancer organization, some of the
functionality and structure may be beneficial for the broader community.
Other packages in the CSI-Cancer organization may depend on this package.

## Structure

Currently, this package contains the main module: `csi_scan_pipeline.py`.
This module contains the abstract base classes for the pipeline, as well as the
`ScanPipeline` class, which is the main class for running the pipeline. This class
contains the `run()` method, which handles parallelization, logging, and passing data
between the module components. 4

## Documentation

For more detailed documentation, open up `docs/index.html` in your browser.

To regenerate the documentation, ensure that you
have [installed the package](#installation) and then run:

```commandline
make_docs
```

## Installation

If you haven't yet, make sure
to [set up an SSH key for GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

1. Activate your `conda` (`conda activate yourenv`) or
   `venv` (`source path/to/your/venv/bin/activate`) environment first.
2. Clone `csi_images` and install:

```commandline
cd ~/path/to/your/repositories
git clone git@github.com:CSI-Cancer/csi_analysis.git
pip install ./csi_analysis
```

Alternatively, you can "editable" install the package, which will allow you to make
changes to the package and have them reflected in your environment without reinstalling:

```commandline
pip install -e ./csi_analysis
```

This will add symbolic links to your `site-packages` directory instead of copying the
package files over.
