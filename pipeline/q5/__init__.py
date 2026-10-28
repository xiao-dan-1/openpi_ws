"""Independent 38-dimensional Q5 data preparation and server-training entrypoints."""

import warnings as _warnings

# The pinned Q5 decoder deliberately uses torchvision's PyAV path. Hide only
# its repeated migration notice; decoding errors and other warnings stay visible.
_warnings.filterwarnings(
    "ignore",
    message=(r"^The video decoding and encoding capabilities of torchvision "
             r"are deprecated from version 0\.22 and will be removed in version 0\.24\."),
    category=UserWarning,
    module=r"^torchvision\.io\._video_deprecation_warning$",
)
